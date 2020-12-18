#!/usr/local/bin/python3
import os
import re
import numpy as np
from tqdm import trange
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import cross_entropy
from graph_trans import GraphEncoder, TransformerConvBlock, ModelConfig, init_weights
rnd = np.random.randn

CUDA = torch.cuda.is_available()
DEVICE = "cpu" if not CUDA else "cuda:0"
VERBOSE = bool(int(os.getenv("VERBOSE", False))) # str to bool always gives True if not ""

import torch
from torch import nn, functional as F

class TextDataset(Dataset):
    def __init__(self, maxlen):
        super().__init__()
        # open a sample text file
        with open("sample.txt", "r", encoding = "utf-8") as f:
            data = f.read()
            data = re.sub(r"[^a-z0-9\s]", "", data.lower())
            vocab = {k:i for i,k in enumerate(sorted(list(set(data))))}
            data = np.array([vocab[x] for x in data])
        # clip data
        data = data[:-(len(data) % (maxlen + 1))]
        data = data.reshape(-1, maxlen + 1)
        self.vocab = vocab
        self.ivocab = {v:k for k,v in self.vocab.items()}
        self.data = data
        self.maxlen = maxlen

        edge_index = []
        for s in range(maxlen):
            for t in range(maxlen - 1, s - 1, -1):
                edge_index.append((s, t))
        edge_index = np.array(edge_index).T
        self.edge_index = edge_index
        self.edge_attr = np.zeros((self.edge_index.shape[1], 18))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        return {
            "input_ids": torch.Tensor(self.data[i, :-1]).long(),
            "sp_tokens": torch.arange(0, self.maxlen, 1).long(),
            "edge_index": torch.Tensor(self.edge_index).long(),
            "edge_attr": torch.Tensor(self.edge_attr).float(),
            "targets": torch.Tensor(self.data[i, 1:]).long(),
        }

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
    
    def forward(self, input_ids, sp_tokens, edge_index, edge_attr, targets = None, verbose = False):
        x = self.wte(input_ids) + self.wpe(sp_tokens)
        x = self.emb_drop(x)
        x, _, _, _ = self.body((x, edge_index, edge_attr, verbose))
        if verbose: print("------>>>>>",x.size())
        logits = self.lm_head(x)
        if verbose: print("@@@@@@@", logits.size())
        out = [logits]
        if targets is not None:
            logits = logits.contiguous().view(-1, logits.size(-1))
            targets = targets.contiguous().view(-1)
            loss = cross_entropy(logits, targets)
            out = out + [loss]
        return out


MAXLEN = 100
data = TextDataset(MAXLEN)
# declare the config
config = ModelConfig(
    vocab_size = len(data.vocab),
    n_embd = 128,
    maxlen = MAXLEN,
    spv_size = MAXLEN,
    n_layer = 4,
    n_head = 8,
    edge_bias = False
)
gpt_model = GPT(config)
gpt_model.apply(init_weights)

print("*** DATASET:", data.data.shape)
print("*** GPT MODEL:", gpt_model.num_params)
# print(data[99])
print("".join([data.ivocab[x] for x in data[99]["input_ids"].numpy()]))

optim = torch.optim.Adam(gpt_model.parameters(), lr = 0.001)
for e in range(10):
    dl = DataLoader(data, batch_size=124, shuffle=True, pin_memory = True if CUDA else False)
    pbar = trange(len(dl))
    for i,d in zip(pbar, dl):
        d = {k:v.to(DEVICE) for k,v in d.items()}
        out = gpt_model(**d, verbose = VERBOSE)
        logits, loss = out
        loss.backward()
        optim.step()
        pbar.set_description(f":: Epoch: {e} :: Iter {i} :: Loss: {loss.item():.3f}")
torch.save(gpt_model.state_dict, "./gptgt.pt")

# generation testing --->

def sample(steps = 10):
    gpt_model.eval()
    seq = torch.Tensor([[data.vocab[x] for x in "Your mom makes sound ".lower()]])
    for _ in trange(steps):
        edge_index = []
        for s in range(seq.size(1)):
            for t in range(seq.size(1) - 1, s - 1, -1):
                edge_index.append((s, t))
        edge_index = torch.Tensor(edge_index).view(1, 2, -1).long()
        # print(i, "----", seq, edge_index.size(-1))
        d = dict(
            input_ids = seq.long(),
            sp_tokens = torch.arange(seq.size(1)).view(1, -1).long(),
            edge_index = edge_index,
            edge_attr = torch.zeros(1, edge_index.size(-1), config.n_embd)
        )
        d = {k:v.to(DEVICE) for k,v in d.items()}
        # print({k:v.size() for k,v in d.items()})
        logits = gpt_model(**d)[0]
        logits = torch.softmax(logits[:, -1, :], dim = -1)
        # print(logits[0])
        tok = torch.argmax(logits[0]).item()
        # print(tok)
        seq = torch.Tensor(seq.tolist()[0] + [tok])
        seq = seq.view(1, seq.size(-1))
    print("".join([data.ivocab[x] for x in seq.tolist()[0]]))

sample(20)
