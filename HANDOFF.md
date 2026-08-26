# HANDOFF — continue building ethikit in Claude Code

This is a working scaffold you can run right now. This document is the context you
need to keep going. Paste it (or open the repo) in Claude Code.

## What ethikit is

An **orchestrator**: a thin framework that wraps mature open-source ethics tools behind
one common interface, runs them over a model+dataset, and standardises the output so it
can be scored, compared, and reported. It does **not** re-implement fairness or
explainability maths — it coordinates the libraries that already do.

## What already works (verified end-to-end, offline)

- Canonical **principle taxonomy** harmonising UNESCO / Microsoft / IBM / Google, with
  provenance and a `tool_addressable` flag (`ethikit/principles.py`).
- **Evaluator interface + registry** — add a tool by subclassing `Evaluator` and calling
  `register()`; nothing else changes (`ethikit/evaluators/base.py`).
- Eight working evaluators: performance (utility), fairness, explainability, robustness,
  membership-inference privacy attack, Giskard scan, ART adversarial robustness, and
  drift detection. Each **prefers a real library and falls back to numpy/sklearn** so the
  demo runs anywhere it can (a few are gated on optional deps -- see `available()`).
- **Model zoo** of stock sklearn classifiers grouped by family (`benchmark/models.py`).
- **Runner** that trains a sweep and runs the battery (`benchmark/runner.py`).
- **Scoring**: min-max normalisation, per-principle profile, Pareto front over
  utility+principles, and an optional weighted ranking (`scoring/aggregate.py`).
- **HTML report** + **Streamlit demo**.
- `python -m examples.run_demo` prints a leaderboard, best-per-family, Pareto front,
  and writes `ethikit_report.html`.

## The four project goals and where each lives

1. **Which tools cover which principle** → `principles.py` (taxonomy) +
   `evaluator.principle_ids` on each evaluator. `principles.coverage_table()` prints it.
2. **What part of the model each tool checks** → `InspectionTarget` on every `EvalResult`
   (`data` / `model_internals` / `predictions` / `behavior` / `process`).
3. **Best model for a dataset + benchmark of "most ethical"** → `benchmark/runner.py` +
   `scoring/aggregate.py` (profile, Pareto, weighted score).
4. **Best model per family, in general** → `benchmark/models.py` families +
   `family_of()`; generalise with the multi-dataset meta-benchmark below (Phase 2).

## First things to do in Claude Code (network available)

1. `pip install -r requirements.txt`
2. `python -m examples.run_demo` — confirm it now pulls the **real UCI Adult** dataset
   (no more synthetic-fallback message) and that fairness reports `backend: fairlearn`.
3. `streamlit run demo/app.py` — click through the UI.

## Roadmap (ML → deep learning)

### Phase 1 — polish the ML demo (you are here)
- Add **AIF360** as a fairness evaluator (70+ metrics; needs Python 3.11) alongside
  fairlearn. Add **SHAP** attributions to the explainability details.
- Replace the `st.line_chart` radar with a real **Plotly radar**.
- Emit a **Google Model Card** per model (Model Card Toolkit) instead of / alongside the
  HTML report — this is the accountability + transparency deliverable.
- Wire in **Giskard** (`pip install giskard`) as a single evaluator: it scans sklearn /
  XGBoost / LightGBM / CatBoost models for bias, robustness, and data leakage in one
  pass, and later extends to LLMs — good long-term backbone.

### Phase 2 — generalisation & "best in general"
- Add **nested cross-validation** (replace single split) and **multi-seed** runs.
- Add a **meta-benchmark** over an OpenML suite (OpenML-CC18) or PMLB so "best per
  family in general" means "aggregated across many datasets," not one.
- Add **statistical significance**: Friedman test + Nemenyi post-hoc (scipy /
  scikit-posthocs) and a critical-difference diagram, so family rankings are defensible.
- Add regression fairness (group MAE/MSE) — `FairnessEvaluator.supports` currently
  classification only.

### Phase 3 — privacy & robustness (deeper) [DONE]
- Swapped the privacy proxy for a real **membership-inference attack**
  (`ethikit/evaluators/privacy.py`): confidence/loss-threshold attack (Yeom et al. 2018)
  with a cross-validated LogisticRegression attack classifier, reporting attack AUC +
  advantage. TensorFlow Privacy's latest release (0.8.11) hard-pins `pandas~=1.4`, which
  has no Python 3.12 wheel and fails to build from source here (confirmed by hand,
  same class of blocker as AIF360) -- its attack algorithm is implemented natively
  against scikit-learn instead of pulling the unmaintained package in. DP-SGD (Opacus)
  not wired in -- no evaluator currently needs a trainable-with-DP model.
- Added the **Adversarial Robustness Toolbox** as a robustness evaluator
  (`ethikit/evaluators/art_robustness.py`): real black-box HopSkipJump evasion attack,
  classification only, tight query/iteration budget (measured by hand against
  RandomForest/GradientBoosting to keep a full model-zoo sweep to ~1-2 minutes).
- Added **drift detection** (`ethikit/evaluators/drift.py`) via Alibi Detect's
  TabularDrift: manufactures a synthetic covariate shift, checks whether the detector
  catches it, and measures the model's own performance drop under it.
- All three tagged `privacy`/`safety` and picked up automatically by
  `ethikit.coverage`'s reverse mapping and every report/Model Card -- no wiring beyond
  registering each evaluator. See `examples/run_phase3_demo.py`.

### Phase 4 — deep learning & LLMs
- The adapter interface already isolates model specifics. For DL: add a **Captum**
  explainability evaluator (integrated gradients etc.) and a PyTorch/TF adapter.
- For LLMs: register **DeepEval** (metrics) + **DeepTeam** (red-teaming) and/or
  **Giskard**'s LLM scan (bias, hallucination, prompt-injection). Principle tags stay
  the same; only the evaluators change.

## Known clean-ups / decisions to revisit
- Performance is tagged with a non-canonical `"utility"` principle id so it forms its own
  axis and isn't double-counted; keep it out of `weights` unless you want to trade utility
  off explicitly.
- The weighted score is deliberately opt-in. Consider adding a **constraint mode**
  ("among models with demographic_parity_diff < 0.1, rank by utility") — often what
  practitioners actually want, and it sidesteps the fairness impossibility problem.
- `data/loaders.py` keeps only numeric columns of Adult for simplicity; add categorical
  encoding for a faithful run.
- Add tests (pytest) around `normalize`, `pareto_front`, and the fallback paths.

## Architecture invariant to preserve
Everything flows as `EvalResult(principle_id, target, metric, value, higher_is_better)`.
As long as a new tool emits that, the rest of the pipeline (scoring, Pareto, report,
UI) works with no changes. Don't let tool-specific output leak past the evaluator.
