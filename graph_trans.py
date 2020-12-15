
import torch
from torch import nn
import torch.nn.functional as F

from torch_scatter import scatter_mean, scatter_max


class TransformerConvBlock(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(config.n_embd, eps = config.layer_norm_epsilon) # for all the inputs
        self.ln2 = nn.LayerNorm(config.n_embd, eps = config.layer_norm_epsilon) # for the edge attributes
        self.lin_q = nn.Linear(config.n_embd, config.n_embd) # linear for query
        self.lin_kv = nn.Linear(config.n_embd, config.n_embd * 2) # linear key value
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

    def collect(self, x, ei):
        return torch.cat([
            x[i, ei[i]] for i in range(ei.size(0))
        ], dim = 0).reshape(ei.size(0), -1, x.size(-1))

    def forward(self, inputs):
        x_orig, edge_index, edge_attr = inputs
        config = self.config
        # [N,dim] -> [E,dim]
        x = self.ln1(x_orig)
        # x = x_orig
        # print(edge_index[:, 0])
        # print(edge_index[:, 1])

        # go from [N,N] --> [E,E] as E < N
        k = self.collect(x, edge_index[:, 0])
        q = self.collect(x, edge_index[:, 1])

        # print(k.size(), q.size())
        # print("---", k.size())
        q = self.lin_q(q)
        e = self.lin_edge(self.ln2(edge_attr))
        q = e + q # update query with edge attr
        # print("q + e:", q.size())
        
        k = self.lin_kv(k)
        k_join, v_join = torch.split(
            tensor = k,
            split_size_or_sections = config.n_embd,
            dim = -1
        )
        q, k, v = self.split_heads(q), self.split_heads(k_join, k = True), self.split_heads(v_join)
        a = self._attn(q, k, v)
        # print("---", x.size(), a.size())
        a = self.merge_heads(a)
        # print("ADFFAFAFAFAa", a)
        hidden = self.ln3(v_join + a) # residual + LN
        a = self.lin_proj(hidden)
        # print(a)
        hidden = a + hidden # residual
        # print("!@#@#@!#$!@#%@#$%^@#$%^3", hidden, edge_index[1])
        hidden = scatter_mean(hidden, edge_index[1], dim = 1)
        # print(edge_index[1], hidden, x.size())
        # pad by the number of indices in input
        if x.size(1) > hidden.size(1):
            # always need to add at last
            size_to_add = (hidden.size(0), x.size(1) - hidden.size(1), x.size(2))
            # print("##---###---###---####", size_to_add)
            hidden = torch.cat([hidden, torch.zeros(size_to_add)], dim = 1)
        # print(hidden.size())
        x = torch.where(hidden != 0., hidden, x)
        # print("#################", x.size())
        return (x, edge_index, edge_attr) # return a list too


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
