
import torch
from torch import nn
import torch.nn.functional as F

from torch_scatter import scatter_mean, scatter_max


class TransformerConvBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(config.n_embd, eps = config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(config.n_embd, eps = config.layer_norm_epsilon)
        self.lin_q = nn.Linear(config.n_embd, config.n_embd)
        self.lin_kv = nn.Linear(config.n_embd, config.n_embd * 2)
        self.lin_edge = nn.Linear(config.n_embd, config.n_embd)
        self.ln3 = nn.LayerNorm(config.n_embd, eps = config.layer_norm_epsilon)
        self.lin_proj = nn.Sequential(*[
            nn.Linear(config.n_embd, config.n_embd * 4),
            nn.ReLU(),
            nn.Linear(config.n_embd * 4, config.n_embd)
        ])
        
    def split_heads(self, x, k = False):
        config = self.config
        new_x_shape = x.size()[:-1] + (config.n_head, x.size(-1) // config.n_head)
        x = x.view(new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1) # [B, H, E, N]
        else:
            return x.permute(0, 2, 1, 3) # [B, H, N, E]
        
    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states
    
    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        w /= (float(v.size(-1)) ** 0.5)
        w = F.softmax(w, dim = -1)
        o = torch.matmul(w, v)
        return o

    def forward(self, inputs):
        x_orig, edge_index, edge_attr = inputs
        config = self.config
        # [N,dim] -> [E,dim]
        x = self.ln1(x_orig)
        k, q = x[..., edge_index[0], :], x[..., edge_index[1], :]
        q = self.lin_q(q)
        e = self.lin_edge(self.ln2(edge_attr))
        q = e + q # update query with edge attr
        
        k = self.lin_kv(k)
        k, v = torch.split(
            tensor = k,
            split_size_or_sections = config.n_embd,
            dim = -1
        )
        q, k, v = self.split_heads(q), self.split_heads(k, k = True), self.split_heads(v)
        a = self._attn(q, k, v)
        print("---", x.size(), a.size())
        x_edge = self.ln3(x + a) # residual + LN
        a = self.lin_proj(x_edge)
        x_edge = a + x_edge # residual
        x_edge = scatter_mean(x_edge, edge_index[1], dim = -2)

        # now how to update the [N,dim] from [E,dim]
        # flow of information is from source to target so target's
        # get updates not the source
        x_orig[..., edge_index[1], :] = x_edge[..., edge_index[1], :]
        
        return (x_orig, edge_index, edge_attr) # return a list too


class GraphEncoder(nn.Module):
    def __init__(self, config, emb):
        super().__init__()
        self.emb = emb  # common embedding across all the models
        self.enc = nn.Sequential(*[
            TransformerConvBlock(config) for _ in range(config.n_layer)
        ])

    @property
    def num_params(self):
        return sum(dict((p.data_ptr(), p.numel()) for p in self.parameters()).values())

    def merge_embeddings(self, node_emb, node_index, req_idx, mode="mean"):
        if mode == "mean":
            embed = scatter_mean(node_emb, node_index, dim=-2)
        elif mode == "max":
            embed = scatter_max(node_emb, node_index, dim=-2)
        return embed[req_idx]

    def forward(self, x, edge_index, edge_attr, node_index=None, req_idx=None):
        emb = self.emb(x)  # get embeddings
        print(emb)
        if node_index is not None and req_idx is not None:
            # merge embeddings if needed
            emb = self.merge_embeddings(x, node_index, req_idx, mode="mean")
        out = self.enc((emb, edge_index, edge_attr))
        return out[0]
