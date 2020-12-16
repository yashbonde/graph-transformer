#!/usr/local/bin/python3
import re
import numpy as np
from types import SimpleNamespace

from torch.nn.functional import cross_entropy
from graph_trans import GraphEncoder, TransformerConvBlock, ModelConfig, init_weights
rnd = np.random.randn

import torch
from torch import nn, functional as F

# open a sample text file
with open("sample.txt", "r", encoding = "utf-8") as f:
    data = f.read()
    data = re.sub(r"[^a-z0-9\s]", "", data.lower())
    vocab = {k:i for i,k in enumerate(sorted(list(set(data))))}
    print(vocab)
    data = np.array([vocab[x] for x in data])

# ---- model ---- #
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.spv_size, config.n_embd)
        self.emb_drop = nn.Dropout(p = 0.1)
        self.body = nn.Sequential(*[
            TransformerConvBlock(config) for _ in range(config.n_layer)
        ])
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.apply(init_weights)

    @property
    def num_params(self):
        return sum(dict((p.data_ptr(), p.numel()) for p in self.parameters()).values())
    
    def forward(self, input_ids, sp_tokens, edge_index, edge_attr, targets = None):
        x = self.wte(input_ids) + self.wpe(sp_tokens)
        x = self.emb_drop(x)
        x, _, _ = self.body((x, edge_index, edge_attr))
        logits = self.lm_head(x)
        out = [logits]
        if targets is not None:
            loss = cross_entropy(logits, targets)
            out = out + [loss]
        return out

# declare the config
config = ModelConfig(
    vocab_size = len(vocab),
    n_embd = 18,
    maxlen = 40,
    spv_size = 40,
    n_layer = 2,
    n_head = 2
)
gpt_model = GPT(config)

# clip data
data = data[:-(len(data) % config.maxlen)]
data = data.reshape(-1, config.maxlen)

print("*** DATASET:", data.shape)
print("*** GPT MODEL:", gpt_model.num_params)

