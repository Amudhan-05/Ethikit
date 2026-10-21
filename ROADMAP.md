# ethikit roadmap: from here to complete

ethikit is a **model-selection and audit tool on ethical principles** -- not a ranker. It runs
locally; nothing is hosted. This plan covers everything still open, in dependency order.

HANDOFF.md's old Phases 2 and 3 are built (cross-validation, significance tests, meta-benchmark,
real membership-inference / adversarial / drift attacks). What follows is what remains *after*
reading the code as it stands today. Each milestone has an exit test: if it isn't true, the
milestone isn't done.

Legend: **S** = under a day, **M** = 1-3 days, **L** = a week or more. "needs net" = needs
network at build time to fetch a dataset (the app itself stays offline-capable).

---

## M1 -- Trust the foundation (tests, CI, honest data)  *(do first; everything else leans on it)*

| # | Work | Size |
|---|---|---|
| 1.1 | `tests/` with pytest: `normalize`, `principle_profile`, `pareto_front`, `weighted_score`; every evaluator on a tiny synthetic dataset (value in range, trace present, `mechanism` non-empty); numpy-fallback vs fairlearn agreement; `prepare_frame` edge cases (NaNs, many-level categoricals, numeric sensitive column, single-class target); checklist scoring | M |
| 1.2 | Console end-to-end test (Playwright, already used manually): a full run, weights re-order without re-run, inspect shows a formula, checklist attaches to the card | M |
| 1.3 | Fix the Adult loader: it keeps **numeric columns only** (`loaders.py:154`); add categorical encoding via `prepare_frame` so the headline real dataset is faithful | S |
| 1.4 | Dependency hygiene: split `requirements.txt` into core / full / dev; document which tools need Python 3.11-3.12 (alibi-detect, model-card-toolkit); make "tool unavailable, here is why" visible in the console and in the model card | S |
| 1.5 | GitHub Actions (tests + lint on 3.11/3.12). Local-first is unchanged: CI is only a checker, nothing is deployed | S |

**Exit:** `pytest` green locally and in CI; Adult run uses categorical features; every skipped tool says why.

## M2 -- Per-domain evidence (answers "does domain matter?")  *(HANDOFF Phase 2, finished properly)*

Today the meta-benchmark is a generic PMLB sample with **no sensitive attribute**, so fairness
never runs in it. Fix that.

| # | Work | Size |
|---|---|---|
| 2.1 | A `domains/` registry: each suite = a domain name, a list of datasets, and for each a declared target, sensitive attribute(s) and positive label. Candidates (availability to verify when built): lending (German Credit, bank marketing, credit card default via fairlearn), income/census (Adult, ACS income), health (hospital readmission via fairlearn `fetch_diabetes_hospital`), criminal-justice (COMPAS, needs a manual CSV), hiring. Datasets that need a manual download get a clear "put the file here" message | L |
| 2.2 | Domain runner: the same sweep per dataset, results tagged `domain`, aggregated **within** a domain first, then (optionally) across domains; Friedman/Nemenyi within a domain; "does the family ordering change across domains?" as an explicit table | M |
| 2.3 | Repeated-CV mode in the console and CLI (`--cv 5x3`): every number becomes a mean +/- spread, drawn on the comparison table and radar; weights re-ordering uses means | M |
| 2.4 | Language audit: "family tendency" claims only appear next to the suite they came from; the field guide's rigor table is regenerated from a real run, not hand-written numbers | S |

**Exit:** `ethikit domains run lending` produces per-domain family tables including fairness
columns; the console can show spread; no doc claims a cross-domain finding the suite can't support.
*Needs net* to build the suites; results are cached on disk afterwards.

## M3 -- Decision support (the "select" half of selection-and-audit)

| # | Work | Size |
|---|---|---|
| 3.1 | **Constraint mode**: "among models with demographic_parity_diff < 0.1 and privacy advantage < 0.2, show best utility" (HANDOFF's own suggestion); thresholds are flags (`--require metric<=value`) and UI controls; failures are listed with the reason, not hidden | M |
| 3.2 | Sensitivity analysis: how much would the weights have to change for the preferred model to change? (a "stability of the choice" panel; protects against over-reading one weight vector) | M |
| 3.3 | Run history and comparison: save runs to a local `runs/` folder, list them, diff two runs (same models, different data/seed/params) | M |
| 3.4 | Save/load run configs (`ethikit run --config run.json`; "export config" in the console) so a demo or audit is reproducible | S |
| 3.5 | Dataset profile step before any model runs: group sizes, base rates per group, missingness, proxy-feature screen (which features predict the sensitive attribute?) -- tool target `data`, currently the thinnest `InspectionTarget` | M |

**Exit:** a user can state requirements, get the feasible set with reasons, see how fragile the
pick is, and re-run the exact same audit from a config file.

## M4 -- Tool depth  *(HANDOFF Phase 3, finished)*

| # | Work | Size |
|---|---|---|
| 4.1 | Fairness: wire fairlearn *mitigation* as an optional "what-if" (ExponentiatedGradient, threshold optimiser): shows the fairness/utility cost of fixing a gap, labelled clearly as a counterfactual, never mixed into the audit scores | M |
| 4.2 | Explainability: SHAP summary in the trace (done partly), add LIME / permutation importance as a second opinion; local explanation for a chosen row in the console | M |
| 4.3 | Privacy: DP-SGD (Opacus) only if/when a deep model arrives (M5); until then document that membership inference is the privacy signal and what it does *not* cover (attribute inference, model extraction) | S |
| 4.4 | Robustness/safety: calibration (ECE) and subgroup performance as tools; keep ART attack budgets configurable and surface the budget in the trace | M |
| 4.5 | Giskard and drift: make both first-class in the console even when optional (clear install hints); confirm drift on Python 3.11/3.12 | S |
| 4.6 | Evaluator contract test: any new tool must supply `formula`, `mechanism`, a `trace`, and an in-range value, enforced by one parametrised test (so adding a tool stays "one file") | S |

**Exit:** every principle that is tool-addressable has at least two independent tools; each
emits a formula, worked steps and an intermediate artefact; mitigation, if present, is
visibly separate from measurement.

## M5 -- Beyond tabular scikit-learn  *(HANDOFF Phase 4)*

| # | Work | Size |
|---|---|---|
| 5.1 | Adapter layer: a thin interface so a model is "anything with predict (+ predict_proba)"; PyTorch and Keras wrappers; sklearn `MLPClassifier` and a small PyTorch MLP join the zoo as a deep-learning family | M |
| 5.2 | Captum attributions (integrated gradients) as the DL explainability tool | M |
| 5.3 | Text/LLM track (separate dataset type, same principle tags): DeepEval metrics, DeepTeam red-teaming, Giskard LLM scan; local models only (no network calls by default); outputs flow into the same `EvalResult` and the same console | L |
| 5.4 | Console: a task switch (tabular / text); the flow graph is unchanged because it already draws "tools grouped by principle" from the registry | M |

**Exit:** a small PyTorch model and a local LLM can be audited through the same console, with
the same principle columns, formulas and model card.

## M6 -- Reporting and governance completeness

| # | Work | Size |
|---|---|---|
| 6.1 | Model Cards: keep the hand-rolled HTML (Model Card Toolkit does not build on Python >=3.12) but also export the Google Model Card JSON schema so a card can be loaded by the real toolkit elsewhere | M |
| 6.2 | Datasheet for the dataset (Gebru et al. sections), pre-filled from the dataset profile (M3.5) and completed by the user, attached to the same audit bundle | M |
| 6.3 | One-click **audit bundle**: a folder or zip with `config.json`, `results.json`, report, model cards, traces, checklist answers, a README of how to reproduce it | M |
| 6.4 | Verify the four checklists against their original texts (NIST AI RMF, EU ALTAI, Deon, Datasheets) and record version/date; flag any paraphrase that drifted | S |
| 6.5 | Print/PDF-friendly model card and report (CSS `@media print`; the reference sheets already do this) | S |

**Exit:** a reviewer can receive one bundle and reproduce or challenge every number in it.

## M7 -- Finish and polish

| # | Work | Size |
|---|---|---|
| 7.1 | Accessibility pass on the console (keyboard-only run, reduced motion, contrast in both themes) | S |
| 7.2 | Update the docs to match reality: proposal (v0.2), Internals, field guide cue cards; README stays current at every milestone | M |
| 7.3 | Packaging: `pip install .` includes the web assets; `ethikit` command works from a clean venv; a `make demo` / one-line start | S |
| 7.4 | Final demo rehearsal script against the field guide, with timings measured on a clean machine | S |

**Exit:** a clean clone follows the README to a working demo with no edits.

---

## Order and parallelism

`M1` first (it protects everything). `M2` and `M3` can run side by side after it (M3.5 feeds
M6.2). `M4` is independent of M2/M3 once M1.1 exists. `M5` is the largest risk, so start it
only after M4.6 (the evaluator contract test) so new tool types can't regress old ones. `M6` and
`M7` close the project.

## Standing rules (apply to every milestone)

- **Local only.** No hosting, no accounts, no telemetry. Network is used only to fetch public datasets, then cached.
- **Selection and audit, not ranking.** No feature may crown a single "best" model; weights and constraints are user inputs and are always shown.
- **README stays current** after every milestone: how to run it locally, what changed, what's new.
- **Personal learning material stays out of git** (`learn/` keeps only general material).
- **Every number is traceable**: formula + worked values + intermediate table, or it doesn't ship.

## Decisions I need from you before starting

1. **Which domains matter for your project?** (lending / income / health / justice / hiring). M2 builds exactly those first.
2. **Is a text/LLM track in scope for your project**, or is deep learning on tabular data enough? (M5.3 is the biggest single item.)
3. **Python version:** are you happy to standardise on 3.12 (everything installs except Model Card Toolkit, which we replace with a JSON export)?
