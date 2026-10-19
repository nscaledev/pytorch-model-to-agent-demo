# pytorch-model-to-agent-demo

Public repo for "From Model to Agent: Running Agentic Coding on Open
Infrastructure", a demo theater session at PyTorch GTC.

See how to go from a deployed model to an agent doing real work. In this live
demo, we deploy an open-weight model on Nscale Dedicated Inference, connect it
to the open-source OpenCode coding agent, and watch it pick up and work through
a real issue. It is a practical demonstration of running agentic coding
workflows on dedicated GPU infrastructure using open tooling, and a hands-on
companion to our breakout session, Lessons at Scale: Building an Open PyTorch
Inference Stack.

## What is in here

The issue the agent works on lives in `namegen`, a tiny character-level GPT
that learns to generate names. The model trains in about ten seconds on a
laptop. A Streamlit page shows the loss curve and freshly sampled names
updating live as it trains, so the audience can see the before and after.

- `namegen/data.py` loads names, builds the character vocab, and makes batches.
- `namegen/model.py` is the GPT: attention, transformer blocks, generation.
- `namegen/train.py` is the training loop, loss estimation, and the CLI.
- `app.py` is the Streamlit UI.
- `tests/` is the pytest suite. It runs in a few seconds.
- `demo/` holds the runbook for the live session.

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
every eval interval and twenty sampled names at the end. Add `?autorun=1` to
the Streamlit URL to start a training run on page load.

## Connecting OpenCode to Nscale

OpenCode reads an `opencode.json` from the repo root. Declare an
OpenAI-compatible provider pointing at your Nscale Dedicated Inference endpoint
and set the model id to whatever you deployed:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "nscale": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Nscale Dedicated Inference",
      "options": {
        "baseURL": "{env:NSCALE_BASE_URL}",
        "apiKey": "{env:NSCALE_API_KEY}"
      },
      "models": {
        "<model-id>": { "name": "<display name>" }
      }
    }
  },
  "model": "nscale/<model-id>"
}
```

Then export the two environment variables and start the agent:

```bash
export NSCALE_BASE_URL="https://<your-dedicated-endpoint>/v1"
export NSCALE_API_KEY="..."
opencode
```

OpenCode also reads an `AGENTS.md` from the repo root for project context such
as the layout, the commands above, and what a healthy training run looks like.
Both files are local to each machine and not tracked in git.
