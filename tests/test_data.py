import torch

from namegen.data import BOUNDARY


def test_vocab_roundtrip(dataset):
    assert dataset.vocab.chars[0] == BOUNDARY
    for w in dataset.train_words[:50]:
        assert dataset.vocab.decode(dataset.vocab.encode(w)) == w


def test_batch_shapes_and_targets(dataset):
    g = torch.Generator().manual_seed(0)
    x, y = dataset.get_batch("train", 16, g)
    assert x.shape == (16, dataset.block_size)
    assert y.shape == (16, dataset.block_size)
    assert x[:, 0].eq(0).all(), "every sequence starts with the boundary token"
    # Targets are x shifted left by one, up to and including the first end boundary.
    for i in range(16):
        n_real = int((y[i] != -1).sum())
        assert n_real >= 2
        assert torch.equal(y[i, : n_real - 1], x[i, 1:n_real])
        assert y[i, n_real - 1] == 0
