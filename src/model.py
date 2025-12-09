# src/model.py
import torch
import torch.nn as nn
from src.gnn_layers import GraphConvolution


class GNNTrafficPredictor(nn.Module):
    def __init__(self, num_nodes, hidden_dim=32, gcn_layers=2, gru_hidden=64, out_window=3):
        super().__init__()

        self.num_nodes = num_nodes
        self.out_window = out_window

        # GCN stack
        self.gcn_layers = nn.ModuleList([
            GraphConvolution(1 if i == 0 else hidden_dim, hidden_dim)
            for i in range(gcn_layers)
        ])

        # GRU takes hidden_dim after GCN
        self.gru = nn.GRU(hidden_dim, gru_hidden, batch_first=True)

        # Output layer for all nodes * predicted time steps
        self.fc = nn.Linear(gru_hidden, num_nodes * out_window)

    def forward(self, x, A):
        """
        x: (B, T, N, 1)
        A: (N, N)
        Returns: (B, N, out_window)
        """
        B, T, N, _ = x.shape

        # -------- Apply GCN at each timestep -------- #
        gcn_out_list = []
        for t in range(T):
            h = x[:, t]              # (B, N, 1)
            out = h
            for gcn in self.gcn_layers:
                out = gcn(out, A)    # (B, N, hidden_dim)
            gcn_out_list.append(out)

        # Stack over time: (B, T, N, hidden_dim)
        gcn_out = torch.stack(gcn_out_list, dim=1)

        # -------- Summarize nodes (mean pooling) -------- #
        gcn_out_mean = gcn_out.mean(dim=2)   # (B, T, hidden_dim)

        # -------- GRU over time -------- #
        gru_out, _ = self.gru(gcn_out_mean)  # (B, T, gru_hidden)

        last_step = gru_out[:, -1]           # (B, gru_hidden)

        # -------- Final FC -------- #
        out = self.fc(last_step)             # (B, N * out_window)
        out = out.view(B, self.num_nodes, self.out_window)

        return out
