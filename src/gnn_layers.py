# src/gnn_layers.py
import torch
import torch.nn as nn

class GraphConvolution(nn.Module):
    """
    Simple GCN layer: H' = ReLU(A_hat @ H @ W)
    Works with batch input by using torch.einsum for stable batching.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, H, A):
        """
        H: (batch, num_nodes, in_features)
        A: (num_nodes, num_nodes) OR (batch, num_nodes, num_nodes)
        returns: (batch, num_nodes, out_features)
        """
        # H @ W  -> (batch, num_nodes, out_features)
        support = torch.einsum("bni,io->bno", H, self.weight)

        # If A is 2D (num_nodes, num_nodes), use it for all batches.
        # Use einsum to multiply: out[b,i,o] = A[i,j] * support[b,j,o]
        if A.dim() == 2:
            out = torch.einsum("ij,bjo->bio", A, support)
        else:
            # A is (batch, N, N)
            out = torch.einsum("bij,bjo->bio", A, support)

        if self.bias is not None:
            out = out + self.bias
        return torch.relu(out)
