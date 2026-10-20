import torch

from namegen.model import CausalSelfAttention, GPT, GPTConfig


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


def test_causal_attention_blocks_information_from_future_positions():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=27, block_size=16, n_layer=1, n_head=2, n_embd=16)
    attn = CausalSelfAttention(cfg).eval()
    x = torch.randn(2, 8, cfg.n_embd)
    with torch.no_grad():
        out = attn(x)
        x_future = x.clone()
        x_future[:, -1, :] += 10.0
        out_future = attn(x_future)
    assert torch.allclose(out[:, :-1], out_future[:, :-1]), (
        "attention output at earlier positions changed when only a future position changed; "
        "the causal mask is not blocking future positions"
    )


def test_model_logits_do_not_depend_on_future_tokens():
    model = small_model()
    torch.manual_seed(1)
    idx = torch.randint(1, 27, (4, 10))
    idx_future = idx.clone()
    idx_future[:, 5:] = idx_future[:, 5:] % 26 + 1
    with torch.no_grad():
        logits, _ = model(idx)
        logits_future, _ = model(idx_future)
    assert torch.allclose(logits[:, :5], logits_future[:, :5]), (
        "logits at earlier positions changed when only future tokens changed; "
        "information is leaking from future positions"
    )
