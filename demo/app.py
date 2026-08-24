"""Streamlit demo UI.

Run (in Claude Code, after `pip install streamlit`):
    streamlit run demo/app.py

Upload nothing needed -- it uses the built-in datasets. Pick model families,
choose per-principle weights, and see the leaderboard, radar profile, and
Pareto front live. This is the seed of the hosted demo (Hugging Face Spaces
runs Streamlit apps for free).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ethikit.data import load_adult, make_synthetic_fairness_dataset
from ethikit.benchmark import run_benchmark, CLASSIFIERS, family_of
from ethikit.scoring import normalize, objective_matrix, pareto_front, weighted_score

st.set_page_config(page_title="Ethikit", layout="wide")
st.title("Ethikit — ethical AI model evaluator")
st.caption("Open-source orchestrator: one dataset in, an ethics profile per model out.")

# --- sidebar controls ------------------------------------------------------
with st.sidebar:
    st.header("1. Data")
    source = st.radio("Dataset", ["Synthetic (biased, offline)", "UCI Adult (needs network)"])
    st.header("2. Models")
    fams = st.multiselect("Model families", list(CLASSIFIERS),
                          default=["linear", "tree_based"])
    st.header("3. Your value weights")
    st.caption("Ethics is multi-objective — you decide what matters most.")
    w = {
        "fairness": st.slider("Fairness", 0.0, 1.0, 0.35),
        "explainability": st.slider("Explainability", 0.0, 1.0, 0.20),
        "safety": st.slider("Safety", 0.0, 1.0, 0.20),
        "privacy": st.slider("Privacy", 0.0, 1.0, 0.15),
        "inclusiveness": st.slider("Inclusiveness", 0.0, 1.0, 0.10),
    }
    go = st.button("Run evaluation", type="primary")

if go:
    ds = (make_synthetic_fairness_dataset() if source.startswith("Synthetic")
          else load_adult())
    factories = {}
    for fam in fams:
        factories.update(CLASSIFIERS[fam])
    if not factories:
        st.warning("Pick at least one model family.")
        st.stop()

    with st.spinner("Training models and running the ethics battery…"):
        raw = run_benchmark(ds, factories)
        dn = normalize(raw)
        obj = objective_matrix(dn)
        obj["weighted_score"] = weighted_score(obj, w).round(3)
        obj = obj.sort_values("weighted_score", ascending=False)
        front = pareto_front(objective_matrix(dn))

    st.subheader("Leaderboard")
    st.caption(f"Dataset: {ds.name} · sensitive attribute: {ds.sensitive_name}")
    st.dataframe(obj.style.background_gradient(cmap="Greens", axis=0), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Per-principle radar")
        radar = objective_matrix(dn).drop(columns=["utility"], errors="ignore")
        st.line_chart(radar.T)  # simple; swap for a true radar with plotly
    with c2:
        st.subheader("Pareto-optimal models")
        for m in front:
            st.markdown(f"- **{m}** ({family_of(m)})")
        st.info("These aren't beaten on every objective at once. "
                "The final pick is a value judgement.")
else:
    st.info("Set options in the sidebar and click **Run evaluation**.")
