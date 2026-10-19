# Demo runbook

Ten minutes total. Target split: deploy and wire up 3 minutes, agent working
5 minutes, wrap 2 minutes.

## Before you walk on

- [ ] Nscale endpoint deployed and warm. Send one request so the first on-stage call is not a cold start.
- [ ] `NSCALE_BASE_URL` and `NSCALE_API_KEY` exported in the terminal you will use.
- [ ] Model id in `opencode.json` matches the deployed model.
- [ ] `gh auth status` is green on this laptop.
- [ ] The repo is on `main` at the buggy commit, working tree clean, issue 1 open on GitHub.
- [ ] Streamlit already running in a browser tab: `.venv/bin/streamlit run app.py`.
- [ ] A fresh `opencode` session already open in the repo so the first prompt costs nothing.
- [ ] Backup recording of a successful run is on the desktop.
- [ ] Terminal font size large enough for the back row.

## Sequence

1. Streamlit tab. Press Train. Point at the loss curve dropping to zero
   and the samples panel full of empty names. "The metric says this model is
   perfect. The samples say it cannot write a single name. This is the issue
   we are going to hand to the agent."
2. Nscale console. Show the deployed model and the endpoint URL.
3. Terminal. Show `opencode.json` and the two env vars.
4. Paste the prompt below. Narrate while the agent reads the code, runs the CLI,
   finds the cause, writes the test, and runs pytest.
5. When the agent opens the PR, switch to GitHub and show the PR body with
   before and after numbers.
6. Back to Streamlit. `git checkout` the PR branch, rerun Train. Loss settles
   around 2.0 and the samples look like names.

## Prompt to paste

```
Pick up issue 1 in this repo: training loss drops to zero but every generated
name is empty. Find the root cause, fix it, add a regression test that would
have caught it, run the full test suite, and open a pull request with before
and after numbers from a CLI training run.
```

## Timing and where to cut

Rehearsal numbers to fill in before the day. Record them per run.

| Phase | Target | Run 1 | Run 2 | Run 3 |
|---|---|---|---|---|
| Read code and run CLI | 45s | | | |
| Root cause identified | 90s | | | |
| Fix plus test written | 60s | | | |
| Tests and CLI rerun | 30s | | | |
| PR opened | 45s | | | |

If the agent is slow in rehearsal, cut from the end of the prompt, not the
start. Drop the PR step first, then the before and after numbers. The fix and
the passing test are the story; the PR is the bow.

If the agent stalls past two minutes without naming the cause, the nudge is:
"Compare the val loss you are seeing with what AGENTS.md says a healthy run
looks like. What would have to be true for a model to hit zero loss?"

If the agent has not found it by four minutes, say so plainly, open the
reference branch diff on screen, and walk through the one-line fix yourself.
Honest beats smooth.

## Reset between rehearsals

```bash
gh pr close <number> --delete-branch   # or leave it and close later
git checkout main && git reset --hard origin/main && git clean -fd
```

Reopen the issue if the agent closed it.

## Reference solution

Branch `reference-fix` contains the one-line fix and the causality test. Diff it
against `main` to check what the agent produced. Do not have it checked out on
stage.
