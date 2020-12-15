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
def test_block():
    set_seed(4)
    gcn = TransformerConvBlock(config)

    # input
    x = torch.Tensor(np.random.randn(1, 4, config.n_embd))
    edge_attr = torch.Tensor(np.random.randn(1, 4, config.n_embd))
    edge_index = torch.Tensor([
        [1, 2, 1, 2],
        [1, 1, 2, 2]
    ]).long()

    # pass through network
    xout, _, _ = gcn((x, edge_index, edge_attr))

    assert torch.all(xout.eq(torch.Tensor([[
        [0.050561707466840744, 0.4999513328075409, -0.9959089159965515, 0.6935985088348389, -0.418301522731781, -1.584577202796936],
        [-0.8502323627471924, 0.3878299593925476, -0.1617470681667328, 0.18628403544425964, 1.4105517864227295, -0.5089848041534424],
        [0.06780585646629333, -0.22946828603744507, -1.2559508085250854, 0.7511060237884521, 0.8013858199119568, 0.30399560928344727],
        [0.7233415842056274, 0.046135567128658295, -0.982991635799408, 0.054432738572359085, 0.15989293158054352, -1.2089481353759766]
    ]]))).item(), "Failed in output value"


def test_network():
    set_seed(4)
    emb = nn.Embedding(config.vocab_size, config.n_embd)
    G = GraphEncoder(config, emb)
    assert G.num_params == 1596, "Failed in number of parameters"

    x = torch.Tensor([[0, 1, 2, 3]]).long()
    edge_attr = torch.Tensor(np.random.randn(1, 4, config.n_embd))
    edge_index = torch.Tensor([
        [1, 2, 1, 2],
        [1, 1, 2, 2]
    ]).long()

    xout = G(x, edge_index, edge_attr)

    assert torch.all(xout.eq(torch.Tensor([[
        [-0.9414165019989014, 1.263246774673462, -0.18376938998699188, 0.1505054533481598, 0.10750222951173782, -0.2780323326587677],
        [-1.501394271850586, 1.9851657152175903, -0.3060815632343292, 0.34965279698371887, 0.26748228073120117, -0.16488206386566162],
        [1.5389409065246582, 0.6355684995651245, 0.1236456036567688, 0.12733393907546997, -1.5070481300354004, -0.9108225107192993],
        [2.3693323135375977, 0.28294411301612854, -0.23453952372074127, 1.6892435550689697, -1.2768363952636719, -1.4579848051071167]
    ]]))).item(), "Failed in output value"
