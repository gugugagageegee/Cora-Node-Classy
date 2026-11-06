import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        dropout=0.5
    ):
        super().__init__()

        self.linear1 = nn.Linear(
            in_channels,
            hidden_channels
        )

        self.linear2 = nn.Linear(
            hidden_channels,
            out_channels
        )

        self.dropout = dropout

    def forward(self, x, edge_index=None):
        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        x = self.linear1(x)
        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        x = self.linear2(x)

        return x


class GCN(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        dropout=0.5
    ):
        super().__init__()

        self.conv1 = GCNConv(
            in_channels,
            hidden_channels
        )

        self.conv2 = GCNConv(
            hidden_channels,
            out_channels
        )

        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        x = self.conv2(x, edge_index)

        return x


class GAT(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        heads=8,
        dropout=0.6
    ):
        super().__init__()

        self.gat1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout
        )

        self.gat2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=dropout
        )

        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        x = self.gat1(x, edge_index)
        x = F.elu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        x = self.gat2(x, edge_index)

        return x