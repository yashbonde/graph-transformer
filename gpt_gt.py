#!/usr/local/bin/python3
import re
import numpy as np
from types import SimpleNamespace
from graph_trans import GraphEncoder, TransformerConvBlock
rnd = np.random.randn

# open a sample text file
with open("sample.txt", "r", encoding = "utf-8") as f:
    data = f.read()
    data = re.sub(r"[^a-z0-9\s]", "", data.lower())
    vocab = {k:i for i,k in enumerate(sorted(list(set(data))))}
    print(vocab)
    data = np.array([vocab[x] for x in data])

# decalre the config
config = SimpleNamespace(
    vocab_size = len(vocab),
    n_embd = 18,
)


