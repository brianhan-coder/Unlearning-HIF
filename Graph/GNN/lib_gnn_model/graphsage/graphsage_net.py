import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import SAGEConv


class SageNet(torch.nn.Module):
    def __init__(self, num_feats, hidden_dim, num_classes):
        super(SageNet, self).__init__()
        
        self.num_layers = 2
        
        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers):
            in_channels = num_feats if i == 0 else hidden_dim
            self.convs.append(SAGEConv(in_channels, hidden_dim))
        
        self.fc = torch.nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)
    
    def forward_once(self, data):
        x = F.relu(self.convs[0](data.x, data.edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs[1](x, data.edge_index)
        x = self.fc(x)
        
        return F.log_softmax(x, dim=-1)
    
    def forward_once_unlearn(self, data):
        x = F.relu(self.convs[0](data.x_unlearn, data.edge_index_unlearn))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs[1](x, data.edge_index)
        x = self.fc(x)
        
        return F.log_softmax(x, dim=-1)
    
    def inference(self, x_all, subgraph_loader, device):
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                x = F.relu(x)
                xs.append(x)
            x_all = torch.cat(xs, dim=0)
        x_all = self.fc(x_all)
        return x_all.cpu()
        