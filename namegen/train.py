"""Training loop and sampling helpers. Run `python -m namegen.train --help`."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Callable

import torch

from .data import NamesDataset
from .model import GPT, GPTConfig


@dataclass
class TrainConfig:
    steps: int = 1000
    batch_size: int = 256
    lr: float = 3e-3
    weight_decay: float = 0.01
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    seed: int = 1337
    device: str = "auto"
    eval_every: int = 50
    eval_batches: int = 4


@dataclass
class StepInfo:
    step: int
    train_loss: float
    val_loss: float | None
    elapsed: float
    tokens_per_sec: float
    samples: list[str] = field(default_factory=list)


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def sample_names(
    model: GPT,
    dataset: NamesDataset,
    n: int = 10,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> list[str]:
    """Sample `n` names from the model, starting from the boundary token."""
    model.eval()
    device = next(model.parameters()).device
    start = torch.zeros(n, 1, dtype=torch.long, device=device)
    out = model.generate(
        start,
        max_new_tokens=dataset.block_size - 1,
        temperature=temperature,
        top_k=top_k,
        generator=generator,
        min_new_tokens=1,  # a name has at least one letter
    )
    names = []
    for row in out.tolist():
        body = row[1:]
        if 0 in body:
            body = body[: body.index(0)]
        names.append(dataset.vocab.decode(body))
    model.train()
    return names


@torch.no_grad()
def estimate_loss(model: GPT, dataset: NamesDataset, cfg: TrainConfig, generator: torch.Generator) -> float:
    model.eval()
    losses = []
    for _ in range(cfg.eval_batches):
        x, y = dataset.get_batch("val", cfg.batch_size, generator)
        _, loss = model(x.to(cfg.device), y.to(cfg.device))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train(
    cfg: TrainConfig,
    dataset: NamesDataset | None = None,
    on_step: Callable[[StepInfo], None] | None = None,
    samples_per_eval: int = 8,
) -> tuple[GPT, NamesDataset, list[StepInfo]]:
    """Train a GPT on the names dataset. Calls `on_step` after every eval interval."""
    torch.manual_seed(cfg.seed)
    cfg.device = pick_device(cfg.device)
    dataset = dataset or NamesDataset()
    gen = torch.Generator().manual_seed(cfg.seed)

    model_cfg = GPTConfig(
        vocab_size=dataset.vocab.size,
        block_size=dataset.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
    )
    model = GPT(model_cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, total_steps=cfg.steps, pct_start=0.1, anneal_strategy="cos"
    )

    history: list[StepInfo] = []
    t0 = time.perf_counter()
    tokens_seen = 0
    model.train()
    for step in range(1, cfg.steps + 1):
        x, y = dataset.get_batch("train", cfg.batch_size, gen)
        x, y = x.to(cfg.device), y.to(cfg.device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        tokens_seen += x.numel()

        if step % cfg.eval_every == 0 or step == cfg.steps:
            elapsed = time.perf_counter() - t0
            info = StepInfo(
                step=step,
                train_loss=loss.item(),
                val_loss=estimate_loss(model, dataset, cfg, gen),
                elapsed=elapsed,
                tokens_per_sec=tokens_seen / elapsed,
                samples=sample_names(model, dataset, n=samples_per_eval, generator=gen),
            )
            history.append(info)
            if on_step:
                on_step(info)

    return model, dataset, history


def main() -> None:
    p = argparse.ArgumentParser(description="Train a tiny character-level GPT on names.")
    p.add_argument("--steps", type=int, default=TrainConfig.steps)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--n-layer", type=int, default=TrainConfig.n_layer)
    p.add_argument("--n-embd", type=int, default=TrainConfig.n_embd)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--device", default="auto")
    p.add_argument("--eval-every", type=int, default=TrainConfig.eval_every)
    p.add_argument("--samples", type=int, default=20, help="names to print at the end")
    args = p.parse_args()

    cfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        seed=args.seed,
        device=pick_device(args.device),
        eval_every=args.eval_every,
    )

    def log(info: StepInfo) -> None:
        print(
            f"step {info.step:5d} | train {info.train_loss:.3f} | val {info.val_loss:.3f} "
            f"| {info.elapsed:5.1f}s | {info.tokens_per_sec/1e3:6.0f}k tok/s | {', '.join(info.samples[:4])}"
        )

    model, dataset, _ = train(cfg, on_step=log)
    print(f"\nmodel: {model.num_params()/1e3:.0f}k params, device={cfg.device}")
    print("\nsamples:")
    gen = torch.Generator().manual_seed(cfg.seed + 1)
    for name in sample_names(model, dataset, n=args.samples, generator=gen):
        print(" ", name)


if __name__ == "__main__":
    main()
