#/usr/local/bin/python3
import random
import numpy as np
from types import SimpleNamespace

import torch
from torch import nn
from graph_trans import TransformerConvBlock, GraphEncoder

# helper functions
def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# global
config = SimpleNamespace(
    n_embd=6,
    layer_norm_epsilon=1e-5,
    n_head=1,
    n_layer=3,
    vocab_size=5,
)

# to check transformer block
def test_block_multi_batch():
    set_seed(4)
    gcn = TransformerConvBlock(config)

    x = torch.Tensor(
        [[[ 0.0506,  0.5000, -0.9959,  0.6936, -0.4183, -1.5846],
         [-0.6477,  0.5986,  0.3323, -1.1475,  0.6187, -0.0880],
         [ 0.4251,  0.3323, -1.1568,  0.3510, -0.6069,  1.5470],
         [ 0.7233,  0.0461, -0.9830,  0.0544,  0.1599, -1.2089]],

        [[ 2.2234,  0.3943,  1.6924, -1.1128,  1.6357, -1.3610],
         [-0.6512,  0.5425,  0.0480, -2.3581, -1.1056,  0.8378],
         [ 2.0879,  0.9148, -0.2762,  0.7965, -1.1438,  0.5099],
         [-1.3475, -0.0094, -0.1307,  0.8021, -0.3030,  1.2020]]])
    edge_attr = torch.Tensor(
        [[[-0.1967,  0.8365,  0.7866, -1.8409,  0.0375,  0.0359],
         [-0.7787,  0.1794, -1.4555,  0.5562,  0.5098,  0.3004],
         [ 2.4766,  0.3523,  0.0675, -0.7323,  0.2971, -0.9618],
         [ 1.2718, -0.6476,  0.1585,  1.9901,  1.1642,  0.2427]],

        [[ 1.3799, -0.0546,  0.7952,  0.0191, -0.9054,  0.4303],
         [ 0.9347, -0.3461, -1.0971, -0.5282, -2.3798, -0.6077],
         [-1.0753,  2.0224, -0.5649, -1.5429,  0.8708, -0.1752],
         [ 0.0486,  0.1886,  0.2093, -0.3744,  0.9547,  0.5232]]])
    edge_index = torch.Tensor([
        [[1, 1, 1, 1], [1, 1, 1, 1]],
        [[1, 1, 1, 2], [0, 3, 2, 3]]
    ]).long()

    with torch.no_grad():
        xout, _, _ = gcn((x, edge_index, edge_attr))
        xout = torch.Tensor(np.round(xout.numpy(), decimals = 3))

    assert torch.all(xout.eq(torch.Tensor(
        [[[ 0.4260,  0.9830, -0.8730,  1.2230, -0.1560, -1.6030],
         [-0.8200, -0.9140,  0.5060,  1.6960, -0.5290, -0.5150],
         [-0.8200, -0.9140,  0.5060,  1.6960, -0.5290, -0.5150],
         [ 1.3690,  0.3660, -1.1570,  0.3790,  0.5350, -1.4910]],

        [[-1.7270, -0.6630, -0.4090,  1.7700,  0.9620,  0.2840],
         [-0.1880,  0.9170,  0.4590, -1.7680, -0.6090,  1.1900],
         [-1.8030, -0.6210, -0.5260,  1.6080,  1.4340,  0.3680],
         [-1.6850, -0.5550, -0.5360,  1.3360,  1.5830,  0.3910]]]
    ))).item(), "Failed in output value"


def test_network():
    set_seed(4)
    emb = nn.Embedding(config.vocab_size, config.n_embd)
    G = GraphEncoder(config)
    assert G.num_params == 1566, "Failed in number of parameters"

    x = torch.Tensor([
        [0, 1, 2, 3, 4],
        [1, 1, 3, 4, 4],
    ]).long()
    edge_attr = torch.Tensor(
        [[[ 0.0506,  0.5000, -0.9959,  0.6936, -0.4183, -1.5846],
         [-0.6477,  0.5986,  0.3323, -1.1475,  0.6187, -0.0880],
         [ 0.4251,  0.3323, -1.1568,  0.3510, -0.6069,  1.5470],
         [ 0.7233,  0.0461, -0.9830,  0.0544,  0.1599, -1.2089],
         [ 2.2234,  0.3943,  1.6924, -1.1128,  1.6357, -1.3610]],

        [[-0.6512,  0.5425,  0.0480, -2.3581, -1.1056,  0.8378],
         [ 2.0879,  0.9148, -0.2762,  0.7965, -1.1438,  0.5099],
         [-1.3475, -0.0094, -0.1307,  0.8021, -0.3030,  1.2020],
         [-0.1967,  0.8365,  0.7866, -1.8409,  0.0375,  0.0359],
         [-0.7787,  0.1794, -1.4555,  0.5562,  0.5098,  0.3004]]])
    edge_index = torch.Tensor([
        [[1, 2, 1, 2, 3], [1, 1, 2, 2, 1]],
        [[1, 1, 1, 2, 2], [0, 3, 3, 2, 1]],
    ]).long()
    x = emb(x)  # get embeddings
    with torch.no_grad():
        xout = G(x, edge_index, edge_attr)
        xout = torch.Tensor(np.round(xout.numpy(), decimals = 3))

    assert torch.all(xout.eq(torch.Tensor(
        [[[-1.4530,  1.8800, -0.3080,  0.1980,  0.1330, -0.4500],
         [ 0.9540, -0.5450,  0.4940, -0.1450, -2.0220,  1.0780],
         [ 1.0310, -0.2620,  0.0760, -0.2450, -1.9300,  1.2730],
         [ 1.5110,  0.0380, -0.3270,  1.0310, -1.0630, -1.1900],
         [-0.3490, -1.1050, -0.6820,  1.2670,  1.4850, -0.6160]],

        [[ 0.8370, -1.2430,  0.5960,  0.1530, -1.6280,  1.1620],
         [ 0.8370, -1.2450,  0.5960,  0.1530, -1.6260,  1.1610],
         [ 0.8370, -1.2450,  0.5960,  0.1530, -1.6260,  1.1610],
         [ 0.8370, -1.2430,  0.5960,  0.1530, -1.6280,  1.1620],
         [-0.3490, -1.1050, -0.6820,  1.2670,  1.4850, -0.6160]]]
    ))).item(), "Failed in output value"
