# namegen

A tiny character-level GPT (about 100k parameters) that learns to generate
human names from `data/names.txt`. Training takes a few seconds on CPU. This
repo exists to demonstrate an agent fixing a real issue end to end.

## Layout

- `namegen/data.py`: loads names, builds the character vocab, packs names into
  fixed-length token sequences, and produces `(x, y)` batches where `y` is `x`
  shifted left by one. Token 0 is the boundary token that starts and ends a name.
- `namegen/model.py`: the GPT. `CausalSelfAttention`, `Block`, `GPT.forward`
  (returns logits and optional cross-entropy loss), and `GPT.generate`.
- `namegen/train.py`: `TrainConfig`, the training loop in `train()`, loss
  estimation, and `sample_names()`. Also the CLI entry point.
- `app.py`: Streamlit UI that runs `train()` and plots loss and samples live.
- `tests/`: pytest suite. Keep it fast. The full suite runs in under 30 seconds.

## Commands

Use the project virtualenv. Create it once with `uv venv && uv pip install -e ".[dev]"`.

```bash
.venv/bin/python -m pytest                      # run the tests
.venv/bin/python -m namegen.train --steps 500    # quick CLI training run, prints loss and samples
.venv/bin/python -m namegen.train                # full run, 1000 steps, under 10 seconds on an Apple Silicon laptop
.venv/bin/streamlit run app.py                   # the UI
```

The CLI prints `step | train loss | val loss | elapsed | throughput | samples`
every eval interval and twenty sampled names at the end. It is the quickest
way to see whether a change helped.

## What good looks like

A healthy 1000-step run ends with train and val loss both around 1.9 to 2.1
and samples that look like plausible names (for example `marisol`, `kaylen`,
`jaxton`). A val loss far below that is a red flag, not a win: the model has
found a shortcut. Loss values that disagree with sample quality mean something
is leaking information between positions or between inputs and targets.

## Conventions

- Plain PyTorch, no training frameworks. Keep the code readable; this is a demo.
- Do not change the public signatures of `train()`, `sample_names()`, or
  `GPT.generate()` without updating `app.py` and the tests.
- Every bug fix ships with a regression test in `tests/` that fails before the
  fix and passes after.
- Run the full test suite before opening a pull request.
- Open pull requests with `gh pr create`. Put the before and after numbers from
  a CLI run in the description.
