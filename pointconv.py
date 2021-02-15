import os.path as osp

import torch
import torch.nn.functional as F
import torch_geometric

import torch_scatter
import typing
from typing import Optional, Callable

def sample_and_knn(pos, x, batch, ratio, k):
  fps_idx = torch_geometric.nn.fps(pos, batch, ratio=ratio)
  fps_idx = torch.sort(fps_idx)[0] 

  sampled_pos = pos[fps_idx] # new_xyz
  sampled_batch = batch[fps_idx]
  knn_edges = torch_geometric.nn.knn(pos, sampled_pos, k, batch_x=batch, batch_y=sampled_batch) # [2, E]

  knn_idx = knn_edges[0].clone()
  knn_edges[0] = fps_idx[knn_edges[0]]
  return knn_edges, fps_idx, knn_idx

@torch.jit.script
def aggr_int(dim_size: int, k: int, node_dim: int, index, new_points, weights, DEVICE: str, density_scale: typing.Optional[torch_geometric.typing.OptTensor] = None):  

  uniq, counts = torch.unique(index, return_counts=True)

  
  # ASSUMPTION: INDEX IS SORTED-> SO UNIQ AND INDEX ARE THE SAME
  # assert torch.all(torch.eq(index, torch.sort(index)[0])) == 1
  
  size_new_points = [dim_size * k, int(list(new_points.size())[-1])]

  arange = torch.arange(counts[0], device=index.device).view(-1, 1)

  prev_i = counts[0]
  arange_dim = 0
  for i in counts[1:]:
    if i == prev_i:
      arange_dim += 1
    else:
      arange = torch.vstack((arange, torch.arange(prev_i, device=index.device).repeat(arange_dim, 1).view(-1, 1)))
      prev_i = i
      arange_dim = 1
  arange = torch.vstack((arange, torch.arange(prev_i, device=index.device).repeat(arange_dim, 1).view(-1, 1)))

  arange = arange.repeat(1, size_new_points[-1])


  index = torch_scatter.utils.broadcast(index, new_points, node_dim)
  idx_clone = (index * k + arange).clone()


  out_new_points = torch.zeros(size_new_points, device=index.device).scatter_(node_dim, idx_clone, new_points).view([-1, k, size_new_points[-1]])

  size_weights = [dim_size * k, int(list(weights.size())[-1])]
  out_weights = torch.zeros(size_weights, device=index.device).scatter_(node_dim, idx_clone[:, :size_weights[-1]], weights).view([-1, k, size_weights[-1]]) 
  
  if density_scale is not None:
    out_new_points *= density_scale.view(-1, 1, 1)

  return out_new_points.permute(0, 2, 1).matmul(out_weights).view([dim_size, -1])


class PointConvSetAbstraction(torch_geometric.nn.MessagePassing):
  def __init__(self, ratio, k, weightnet=None, local_nn = None, global_nn = None):
    super(PointConvSetAbstraction, self).__init__(aggr=None, flow='target_to_source')
    self.ratio = ratio
    self.k = k
    self.local_nn = local_nn
    self.global_nn = global_nn
    self.weightnet = weightnet

  def forward(self, pos, x, batch):
    # pos: [N, P]
    # x: [N, F]
    # edge_index: [2, E]
    # batch: [N]
    knn_edges, fps_idx, knn_idx = sample_and_knn(pos, x, batch, self.ratio, self.k)

    out = self.propagate(knn_edges, x=x, pos=pos, batch=batch, knn_idx=knn_idx, fps_idx=fps_idx, fps_idx_shape=fps_idx.shape[0])

    if self.global_nn is not None:
      out = self.global_nn(out)

    return out, fps_idx

  def message(self, x, edge_index, pos_i, pos_j, x_j):
    grouped_norm = (pos_j - pos_i)

    if x_j is not None:
      msg =  torch.cat([grouped_norm.clone(), x_j], dim=1)
    else:
      msg = grouped_norm.clone()

    if self.weightnet is not None:
      grouped_norm = self.weightnet(grouped_norm)

    if self.local_nn is not None:
      return self.local_nn(msg), msg, grouped_norm

    return msg, msg, grouped_norm


  def aggregate(self, msg_output, knn_idx, fps_idx_shape, density_scale=None):
    new_points, msg, weights = msg_output # weights: nn(msg), msg: new_points, grouped_norm: grouped_norm
    return aggr_int(fps_idx_shape, self.k, self.node_dim, knn_idx, new_points, weights, DEVICE, density_scale=density_scale)


class DensityNet(torch_geometric.nn.MessagePassing):
  def __init__(self, k=100, nn=None, kernel="gaussian", kernel_params=None):
    super(DensityNet, self).__init__(aggr='mean', flow='target_to_source')
    self.k = k
    self.nn = nn
    self.kernel = kernel
    self.kernel_params = kernel_params

  def forward(self, pos, batch, fps_idx):
    knn_edges = torch_geometric.nn.knn_graph(pos, k=self.k, batch=batch, flow=self.flow)
    
    pos_density = self.propagate(knn_edges, pos=pos)
    inv_density = 1/(pos_density + 1e-10)

    sampled_inv_density = inv_density[fps_idx]

    if self.nn is not None:
      sampled_inv_density = self.nn(sampled_inv_density)

    return sampled_inv_density

  def message(self, pos_i, pos_j, edge_index):
    sq_dist = torch.norm(pos_j - pos_i, dim = 1, keepdim=True)
    
    if self.kernel == "gaussian":
      bandwidth = self.kernel_params[0]
      density = torch.exp(-sq_dist / (2.0 * bandwidth * bandwidth)) / (2.5 * bandwidth)
    else:
      raise NotImplementedError

    return density

class PointConvDensitySetAbstraction(PointConvSetAbstraction):
  def __init__(self, ratio, k, weightnet=None, local_nn = None, global_nn = None, density_nn=None, kernel_params=None):
    super(PointConvDensitySetAbstraction, self).__init__(ratio, k, weightnet=weightnet, local_nn=local_nn, global_nn=global_nn)
    self.density_net = DensityNet(nn=density_nn, kernel_params=kernel_params)    

  def aggregate(self, msg_output, knn_idx, pos, batch, fps_idx, fps_idx_shape):    
    density_scale = self.density_net.forward(pos, batch, fps_idx) 
    return super(PointConvDensitySetAbstraction, self).aggregate(msg_output, knn_idx, fps_idx_shape, density_scale=density_scale)

class PointNetPool(torch.nn.Module):
  def __init__(self, nn):
    super(PointNetPool, self).__init__()
    self.nn = nn

  def forward(self, x, pos, batch):
    x = self.nn(torch.cat([x, pos], dim=1))
    x = torch_geometric.nn.global_max_pool(x, batch)
    return x


def mlp(channels, batch_norm=True):
    return torch.nn.Sequential(*[
        torch.nn.Sequential(torch.nn.Linear(channels[i - 1], channels[i]), torch.nn.ReLU(), torch.nn.BatchNorm1d(channels[i]))
        for i in range(1, len(channels))
    ])



class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = PointConvDensitySetAbstraction(0.5, 32,
                                                    weightnet=mlp([3, 8, 4]),
                                                    local_nn=mlp([6, 16, 16]),
                                                    global_nn=mlp([4*16, 256, 256]),
                                                    density_nn=mlp([1, 8, 1]),
                                                    kernel_params=0.2)

        self.conv2 = PointConvDensitySetAbstraction(0.25, 64,
                                                    weightnet=mlp([3, 8, 4]),
                                                    local_nn=mlp([256 + 3, 256, 256]),
                                                    global_nn=mlp([1024, 1024, 2048]),
                                                    density_nn=mlp([1, 8, 1]),
                                                    kernel_params=0.2)

        self.pool = PointNetPool(mlp([3 + 2048, 2048, 2048]))

        self.lin1 = torch.nn.Linear(2048, 512)
        self.lin2 = torch.nn.Linear(512, 256)
        self.lin3 = torch.nn.Linear(256, 10)
    
    def forward(self, data):
        pos = data.pos
        out = data.x
        batch = data.batch
        
        out, fps_idx = self.conv1(pos, out, batch)
        pos = pos[fps_idx]
        batch = batch[fps_idx]

        out, fps_idx = self.conv2(pos, out, batch)
        pos = pos[fps_idx]
        batch = batch[fps_idx]

        out = self.pool(out, pos, batch)

        x = F.relu(self.lin1(out))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)

        return F.log_softmax(x, dim=-1)

        

def train(epoch, optimizer):
    model.train()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        optimizer.step()

def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

if __name__ == "__main__":
    path = osp.join('.', 'data/ModelNet10')
    
    pre_transform, transform = torch_geometric.transforms.NormalizeScale(), torch_geometric.transforms.SamplePoints(1024)

    train_dataset = torch_geometric.datasets.ModelNet(path, '10', True, transform, pre_transform)
    test_dataset = torch_geometric.datasets.ModelNet(path, '10', False, pre_transform, transform)

    train_loader = torch_geometric.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch_geometric.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 201):
      train(epoch, optimizer)
      test_acc = test(test_loader)
      print('Epoch: {:03d}, Test: {:.4f}'.format(epoch, test_acc))