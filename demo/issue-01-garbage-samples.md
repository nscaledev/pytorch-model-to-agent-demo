Title: Training loss collapses to zero but generated names are a single letter repeated

**What happens**

Running `python -m namegen.train` the loss falls to 0.02 within a hundred steps
and reaches 0.000 by step 500. Val loss agrees with it. Every sampled name is
one character repeated to the maximum length:

```
step   100 | train 0.024 | val 0.020 |   1.2s |    328k tok/s | iiiiiiiiiiiiiii, ppppppppppppppp, iiiiiiiiiiii, hhhhhhhhhhhhh
step   500 | train 0.000 | val 0.000 |   4.9s |    421k tok/s | iiiiiiiiiiiiiii, iiiiiiiiiiiiiii, rrrrrrrrrrrr, iiiiiiiiiiii
step  1000 | train 0.000 | val 0.000 |   9.2s |    443k tok/s | iiiiiiiiiiiiiii, pppppppppppppww, iiiiiiiiiiiii, iiiiiiiiiiiiii
```

The Streamlit app shows the same thing: the loss chart dives straight to the
floor and the samples panel is a column of repeated letters.

**What I expect**

AGENTS.md says a healthy run finishes around a loss of 2.0 with samples that
look like real names. A character-level model over 27 symbols cannot honestly
reach a loss of 0.000, so something is letting the model cheat during training
in a way it cannot repeat at generation time.

**Ask**

- Find the root cause and fix it.
- Add a regression test that fails on the current `main` and passes with the fix.
- Run the full test suite.
- Open a pull request with before and after numbers from a CLI run.
