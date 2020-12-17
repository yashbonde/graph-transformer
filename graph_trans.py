
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

    def _attn(self, q, k, v,):
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
    def __init__(self, config):
        super().__init__()
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
        if node_index is not None and req_idx is not None:
            # merge embeddings if needed
            x = self.merge_embeddings(x, node_index, req_idx, mode="mean")
        out = self.enc((x, edge_index, edge_attr))
        return out[0]


class ModelConfig():
    n_embd = None
    layer_norm_epsilon=1e-5 # epsilon for layer norm
    initializer_range=0.2
    n_head = None # number of heads
    n_layer = None # number of layers
    vocab_size = None # vocab size of primary tokens
    spv_size = None # special position tags (secondary tokens)
    def __init__(self, **kwargs):
        self.attrs = []
        for k,v in kwargs.items():
            setattr(self, k, v)
            self.attrs.append(k)

    def __repr__(self):
        return "---- MODEL CONFIGURATION ----\n" + \
            "\n".join([
                f"{k}\t{getattr(self, k)}" for k in list(set(self.attrs))
            ]) + "\n"


# --- helper functions --- #
def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.2)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, (nn.LayerNorm)):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

def configure_optimizers(model, train_config):
    """
    from karpathy/minGPT
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding) # add denorm here
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            
            if "ValueHead" in fpn: # no decay for value head layers
                no_decay.add(fpn)

            pn_type = pn.split(".")[-1]
            if pn_type == 'bias':
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn_type == 'weight' and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn_type == 'weight' and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=train_config.lr, betas=train_config.betas)
    return optimizer
