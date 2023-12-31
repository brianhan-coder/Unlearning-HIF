import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv
import pdb

class GINNet(torch.nn.Module):
    def __init__(self, num_feats, num_classes):
        super(GINNet, self).__init__()

        dim = 32
        self.num_layers = 2

        nn1 = Sequential(Linear(num_feats, dim), ReLU(), Linear(dim, dim))
        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))

        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(nn1))
        self.convs.append(GINConv(nn2))

        self.bn = torch.nn.ModuleList()
        self.bn.append(torch.nn.BatchNorm1d(dim))
        self.bn.append(torch.nn.BatchNorm1d(dim))

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, num_classes)

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)

            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

            x = self.bn[i](x)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

    def forward_once(self, data):
        x = F.relu(self.convs[0](data.x, data.edge_index))
        x = self.bn[0](F.dropout(x, p=0.5, training=self.training))
        x = self.convs[1](x, data.edge_index)
        x = self.bn[1](x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
    
    def forward_once_unlearn(self, data):
        x = F.relu(self.convs[0](data.x_unlearn, data.edge_index_unlearn))
        
        x = self.bn[0](F.dropout(x, p=0.5, training=self.training))
        
        x = self.convs[1](x, data.edge_index_unlearn)
        pdb.set_trace()
        x = self.bn[1](x)
        if torch.all(torch.isnan(x)) or torch.all(torch.isinf(x)):
            print("\033[31m [forward_once_unlearn] 1 has NAN value \033[0m")
            pdb.set_trace()
        x = F.relu(self.fc1(x))
        if torch.all(torch.isnan(x)) or torch.all(torch.isinf(x)):
            print("\033[31m [forward_once_unlearn] 2 has NAN value \033[0m")
            pdb.set_trace()
        x = F.dropout(x, p=0.5, training=self.training)
        if torch.all(torch.isnan(x)) or torch.all(torch.isinf(x)):
            print("\033[31m [forward_once_unlearn] 3 has NAN value \033[0m")
            pdb.set_trace()
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

    def inference(self, x_all, subgraph_loader, device):
        for i in range(self.num_layers):
            xs = []

            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)

                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)

                if i != self.num_layers - 1:
                    x = F.relu(x)

                x = self.bn[i](x)

                xs.append(x)

            x_all = torch.cat(xs, dim=0)

        x_all = F.relu(self.fc1(x_all))
        x_all = self.fc2(x_all)

        return x_all.cpu()
