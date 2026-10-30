import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, output, dropout, edge_dim, num_hidden_layers=6):
        super(GCN, self).__init__()

        # Initial layer to transform the input features from nfeat to nhid
        self.first_layer = nn.Linear(nfeat, nhid)

        # Create an iterable list of graph convolution layers based on num_hidden_layers
        self.convs = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.convs.append(GINEConv(
                nn=nn.Sequential(
                    nn.Linear(nhid, nhid),
                    nn.ReLU(),
                    nn.Linear(nhid, nhid)
                ), 
                edge_dim=edge_dim
            ))
        
        # Dropout for regularization
        self.dropout = dropout

        # Fully connected layers for final prediction
        self.lin1 = nn.Linear(nhid, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, output)

    def forward(self, graph, is_training=True):
        # Build the graph (this function is assumed to exist in your environment)
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
        
        # Ensure the proper data types for input to the convolution layers
        assert edge_index.dtype == torch.long
        assert edge_attr.dtype == torch.float

        # Apply the first layer to transform input features
        x = F.relu(self.first_layer(x))

        # Apply graph convolution layers iteratively with ReLU and dropout
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_attr))
            x = F.dropout(x, self.dropout, training=self.training)

        # Apply fully connected layers for final prediction
        x = F.relu(self.lin1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.lin2(x))
        x = self.lin3(x)  # Final output layer
        
        return x
