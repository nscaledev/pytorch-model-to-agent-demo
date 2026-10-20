import torch

from namegen.model import GPT, GPTConfig


def small_model(vocab_size=27, block_size=16):
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=vocab_size, block_size=block_size, n_layer=1, n_head=2, n_embd=16))


def test_forward_shapes():
    model = small_model()
    idx = torch.randint(0, 27, (4, 10))
    logits, loss = model(idx)
    assert logits.shape == (4, 10, 27)
    assert loss is None


def test_loss_with_targets_ignores_padding():
    model = small_model()
    idx = torch.randint(0, 27, (4, 10))
    targets = torch.randint(0, 27, (4, 10))
    targets[:, 5:] = -1
    _, loss = model(idx, targets)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_attention_mask_blocks_information_from_future_positions():
    model = small_model()
    model.eval()
    idx = torch.randint(0, 27, (2, 10))
    with torch.no_grad():
        base_logits = model(idx)[0]
    idx[:, 6:] = (idx[:, 6:] + 7) % 27  # change only future tokens
    with torch.no_grad():
        changed_logits = model(idx)[0]
    assert torch.allclose(base_logits[:, :6], changed_logits[:, :6]), (
        "logits at earlier positions changed when only future tokens were altered; "
        "attention is not causal"
    )


def test_generate_extends_sequence():
    model = small_model()
    start = torch.zeros(3, 1, dtype=torch.long)
    out = model.generate(start, max_new_tokens=8)
    assert out.shape == (3, 9)
    assert out[:, 0].eq(0).all()


def test_generate_min_new_tokens_blocks_end_token():
    model = small_model()
    start = torch.zeros(8, 1, dtype=torch.long)
    out = model.generate(start, max_new_tokens=3, min_new_tokens=2)
    assert (out[:, 1:3] != 0).all()
