# namegen

A tiny character-level GPT that learns to generate names, used in the
"From Model to Agent: Running Agentic Coding on Open Infrastructure" demo at
PyTorch GTC. The model trains in about ten seconds on a laptop. A Streamlit
page shows the loss curve and freshly sampled names updating live as it trains.

The demo deploys an open-weight model on Nscale Dedicated Inference, points the
open-source OpenCode coding agent at it, and lets the agent pick up an issue in
this repo, work through it, and open a pull request.

## Setup

```bash
uv venv && uv pip install -e ".[dev]"
mkdir -p data && curl -sSL -o data/names.txt \
  https://raw.githubusercontent.com/karpathy/makemore/master/names.txt
```

The names list is 32k names from Andrej Karpathy's makemore, originally from
US Social Security baby name data. It is not tracked in git.

## Run

```bash
.venv/bin/python -m namegen.train --steps 500   # CLI training, prints loss and samples
.venv/bin/streamlit run app.py                   # live UI
.venv/bin/python -m pytest                       # tests
```

The CLI prints `step | train loss | val loss | elapsed | throughput | samples`
every eval interval and twenty sampled names at the end.

## Layout

- `namegen/data.py` loads names, builds the character vocab, and makes batches.
- `namegen/model.py` is the GPT: attention, transformer blocks, generation.
- `namegen/train.py` is the training loop, loss estimation, and the CLI.
- `app.py` is the Streamlit UI.
- `tests/` is the pytest suite. It runs in a few seconds.
- `demo/` holds the runbook for the live session.

## Connecting OpenCode to Nscale

`opencode.json` declares an OpenAI-compatible provider called `nscale`. Set two
environment variables and change the model id to whatever you deployed:

```bash
export NSCALE_BASE_URL="https://<your-dedicated-endpoint>/v1"
export NSCALE_API_KEY="..."
opencode
```

OpenCode reads an `AGENTS.md` from the repo root for project context such as
the layout, the commands above, and what a healthy training run looks like.
That file is local to each machine and not tracked in git.
