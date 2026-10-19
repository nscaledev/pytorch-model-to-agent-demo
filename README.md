# namegen

A tiny character-level GPT that learns to generate names, used in the
"From Model to Agent: Running Agentic Coding on Open Infrastructure" demo.
The model trains in a few seconds on a laptop CPU. A Streamlit page shows the
loss curve and freshly sampled names updating live as it trains.

The demo deploys an open-weight model on Nscale Dedicated Inference, points the
open-source OpenCode agent at it, and lets the agent pick up an issue in this
repo, fix it, and open a pull request.

## Quick start

```bash
uv venv && uv pip install -e ".[dev]"
.venv/bin/python -m namegen.train --steps 500   # CLI training, prints samples
.venv/bin/streamlit run app.py                   # live UI
.venv/bin/python -m pytest                       # tests
```

## Connecting OpenCode to Nscale

`opencode.json` declares an OpenAI-compatible provider called `nscale`. Set two
environment variables and change the model id to whatever you deployed:

```bash
export NSCALE_BASE_URL="https://<your-dedicated-endpoint>/v1"
export NSCALE_API_KEY="..."
opencode
```

`AGENTS.md` gives the agent the repo map, the commands, and what a healthy run
looks like. See `demo/RUNBOOK.md` for the on-stage sequence and reset steps.

## Data

`data/names.txt` is the 32k names list from Andrej Karpathy's makemore,
originally from US Social Security baby name data.
