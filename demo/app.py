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
import plotly.graph_objects as pgo
import streamlit as st

from ethikit.data import load_adult, make_synthetic_fairness_dataset
from ethikit.benchmark import run_benchmark, CLASSIFIERS, family_of
from ethikit.scoring import normalize, objective_matrix, pareto_front, principle_profile, weighted_score
from ethikit.report import render_model_cards

def plotly_radar(radar: pd.DataFrame) -> pgo.Figure:
    """One closed polygon per model over the per-principle profile columns."""
    categories = list(radar.columns)
    theta = categories + [categories[0]]  # close the loop
    fig = pgo.Figure()
    for model in radar.index:
        r = radar.loc[model].fillna(0.0).tolist()
        fig.add_trace(pgo.Scatterpolar(
            r=r + [r[0]], theta=theta, name=model, fill="toself", opacity=0.55,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True, margin=dict(l=40, r=40, t=20, b=20), height=420,
    )
    return fig


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
        st.plotly_chart(plotly_radar(radar), use_container_width=True)
    with c2:
        st.subheader("Pareto-optimal models")
        for m in front:
            st.markdown(f"- **{m}** ({family_of(m)})")
        st.info("These aren't beaten on every objective at once. "
                "The final pick is a value judgement.")

    st.subheader("Model cards")
    st.caption("Accountability + transparency deliverable, one per model "
              "(model details, quantitative analysis, ethical considerations, caveats).")
    picked = st.selectbox("Model", list(obj.index))
    if picked:
        card_paths = render_model_cards(
            ds.name, raw, principle_profile(dn), front,
            out_dir="model_cards", sensitive_name=ds.sensitive_name,
        )
        card_path = next(p for p in card_paths if p.endswith(f"{picked}.html"))
        with open(card_path, encoding="utf-8") as fh:
            card_html = fh.read()
        st.components.v1.html(card_html, height=900, scrolling=True)
        st.download_button("Download this model card (HTML)", card_html,
                           file_name=f"{picked}_model_card.html", mime="text/html")
else:
    st.info("Set options in the sidebar and click **Run evaluation**.")
