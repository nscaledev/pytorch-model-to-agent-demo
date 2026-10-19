"""Streamlit UI: train the tiny GPT live and watch loss and generated names update."""

from __future__ import annotations

import pandas as pd
import streamlit as st
import torch

from namegen.data import NamesDataset
from namegen.train import StepInfo, TrainConfig, pick_device, sample_names, train

st.set_page_config(page_title="namegen", page_icon="🔤", layout="wide")
st.title("namegen: a tiny character-level GPT")
st.caption("Trains a 2-layer transformer on 32k names in a few seconds. Watch the loss fall and the samples improve.")


@st.cache_resource(show_spinner=False)
def load_dataset() -> NamesDataset:
    return NamesDataset()


dataset = load_dataset()

with st.sidebar:
    st.header("Training")
    steps = st.slider("Steps", 200, 3000, 1000, step=100)
    batch_size = st.select_slider("Batch size", [64, 128, 256, 512], value=256)
    lr = st.select_slider("Peak learning rate", [1e-3, 2e-3, 3e-3, 5e-3, 1e-2], value=3e-3)
    n_layer = st.slider("Layers", 1, 4, 2)
    n_embd = st.select_slider("Embedding dim", [32, 64, 128], value=64)
    seed = st.number_input("Seed", value=1337, step=1)
    eval_every = st.slider("Eval every N steps", 25, 200, 50, step=25)
    st.divider()
    st.header("Sampling")
    temperature = st.slider("Temperature", 0.5, 1.5, 1.0, step=0.1)
    top_k = st.slider("Top-k (0 = off)", 0, 27, 0)
    st.divider()
    st.caption(f"{len(dataset.train_words):,} train / {len(dataset.val_words):,} val names")
    st.caption(f"vocab {dataset.vocab.size} · block size {dataset.block_size}")

run = st.button("Train", type="primary", use_container_width=True)

# Visiting /?autorun=1 starts one training run on page load (handy for demos and screenshots).
if st.query_params.get("autorun") == "1" and not st.session_state.get("autorun_done"):
    st.session_state["autorun_done"] = True
    run = True

stat_cols = st.columns(4)
stat_step = stat_cols[0].empty()
stat_train = stat_cols[1].empty()
stat_val = stat_cols[2].empty()
stat_tps = stat_cols[3].empty()

left, right = st.columns([3, 2])
with left:
    st.subheader("Loss")
    chart_slot = st.empty()
with right:
    st.subheader("Samples")
    samples_slot = st.empty()


def render_stats(info: StepInfo) -> None:
    stat_step.metric("Step", f"{info.step}")
    stat_train.metric("Train loss", f"{info.train_loss:.3f}")
    stat_val.metric("Val loss", f"{info.val_loss:.3f}" if info.val_loss is not None else "–")
    stat_tps.metric("Throughput", f"{info.tokens_per_sec/1e3:.0f}k tok/s", help=f"{info.elapsed:.1f}s elapsed")


def render_samples(names: list[str], title: str = "") -> None:
    body = "\n".join(f"- `{n or '∅'}`" for n in names)
    samples_slot.markdown(f"{title}\n\n{body}" if title else body)


if run:
    cfg = TrainConfig(
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        n_layer=n_layer,
        n_embd=n_embd,
        seed=int(seed),
        device=pick_device("auto"),
        eval_every=eval_every,
    )
    rows: list[dict] = []

    def on_step(info: StepInfo) -> None:
        rows.append({"step": info.step, "train": info.train_loss, "val": info.val_loss})
        df = pd.DataFrame(rows).set_index("step")
        chart_slot.line_chart(df, height=360)
        render_stats(info)
        render_samples(info.samples, title=f"step {info.step}")

    with st.spinner("Training…"):
        model, _, history = train(cfg, dataset=dataset, on_step=on_step, samples_per_eval=10)

    st.session_state["model"] = model
    st.session_state["history"] = history
    st.session_state["rows"] = rows
    st.session_state.pop("generated", None)

elif "history" in st.session_state:
    # Streamlit reruns this script on every interaction. Redraw the last run so the
    # chart and samples survive clicks elsewhere on the page.
    last = st.session_state["history"][-1]
    chart_slot.line_chart(pd.DataFrame(st.session_state["rows"]).set_index("step"), height=360)
    render_stats(last)
    render_samples(last.samples, title=f"step {last.step}")

if "model" in st.session_state:
    model = st.session_state["model"]
    history = st.session_state["history"]
    st.success(
        f"Done: {history[-1].elapsed:.1f}s, {model.num_params()/1e3:.0f}k params, final val loss {history[-1].val_loss:.3f}"
    )
    st.divider()
    if st.button("Generate 20 more names"):
        gen = torch.Generator().manual_seed(torch.seed() % (2**31))
        st.session_state["generated"] = sample_names(
            model, dataset, n=20, temperature=temperature, top_k=top_k or None, generator=gen
        )
    if "generated" in st.session_state:
        cols = st.columns(4)
        for i, n in enumerate(st.session_state["generated"]):
            cols[i % 4].code(n or "∅ (empty)")
