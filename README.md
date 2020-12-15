# graph-transformer

Transformer Network at its core is a graph network where the information move from source node to target node. All the operations can be defined with this format
```

BERT: (Fully connected Network), the adjacency matrix can be given as:
[[1, 1, 1],
 [1, 1, 1],
 [1, 1, 1]]

GPT: (Forward Connected Network), the adjacency matrix can be given as:
    A' B' C'
A [[1, 0, 0],
B  [1, 1, 0],
C  [1, 1, 1]]

which say that information for A comes from A, information for B comes from A and B. In this case A
is token at t, B (t+1) and C (t+2).
```

Using the above approach we can then define any transformer block with indexing.

### Tests

To run the tests, you must have `pytest` installed on your system, run `pytest` in CLI. To play with the GPT implementation run `gpt_gt.py` (currently building)!

### Samples

These will be the text samples build out GPT model.


## Credits

Code is under MIT License and the `sample.txt` is from [here](https://www.unqualified-reservations.org/2009/01/gentle-introduction-to-unqualified/).
