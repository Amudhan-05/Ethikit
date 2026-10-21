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

## M1 -- Trust the foundation (tests, CI, honest data)  **[DONE]**

| # | Work | Size |
|---|---|---|
| 1.1 | `tests/` with pytest: `normalize`, `principle_profile`, `pareto_front`, `weighted_score`; every evaluator on a tiny synthetic dataset (value in range, trace present, `mechanism` non-empty); numpy-fallback vs fairlearn agreement; `prepare_frame` edge cases (NaNs, many-level categoricals, numeric sensitive column, single-class target); checklist scoring | M |
| 1.2 | Console end-to-end test (Playwright, already used manually): a full run, weights re-order without re-run, inspect shows a formula, checklist attaches to the card | M |
| 1.3 | Fix the Adult loader: it keeps **numeric columns only** (`loaders.py:154`); add categorical encoding via `prepare_frame` so the headline real dataset is faithful | S |
| 1.4 | Dependency hygiene: standardise on Python 3.12 (`.python-version`, README, CI); split `requirements.txt` into core / full / dev; make "tool unavailable, here is why" visible in the console and in the model card | S |
| 1.5 | GitHub Actions (tests + lint on 3.11/3.12). Local-first is unchanged: CI is only a checker, nothing is deployed | S |

**Exit:** `pytest` green locally and in CI on Python 3.12; Adult run uses categorical features; every skipped tool says why.

## M2 -- Reference suites per domain (choose a model *before* you have your own data)  **[BUILT -- live-download check pending]**

**Purpose.** When an organisation has no trained model or labelled data yet, the honest question
is: "for data like ours, which kind of model tends to do best on the ethics profile, so where
should we start?" A **reference suite** is a curated set of public datasets from one domain,
each with a declared target and sensitive attribute, run through the same battery. The result
is a *prior* (a starting shortlist with evidence), never a verdict -- it has to be re-checked
on the organisation's own data as soon as that exists. Today's meta-benchmark can't do this: it
is a generic PMLB sample with **no sensitive attribute**, so fairness never runs in it.

| # | Work | Size |
|---|---|---|
| 2.1 | `domains/` registry: a suite = domain name + datasets, each with target, sensitive attribute(s), positive label, licence/source note. Loaders cache to a local folder; manual-download datasets give a clear "put the file here" message | M |
| 2.2 | Suites to build (availability checked, see table below) | L |
| 2.3 | Domain runner: the same sweep per dataset, tagged `domain`/`dataset`, aggregated **within** a domain; Friedman/Nemenyi within a domain; an explicit "does the family ordering change across domains?" table | M |
| 2.4 | Console + CLI: pick a domain ("I'm a lender"), see the family-level evidence per principle with spread across its datasets, and carry the shortlist into a normal run on your own data. Always shows suite size, datasets and the caveats | M |
| 2.5 | Repeated-CV mode (`--cv 5x3`) so every number carries a spread; the radar and tables show it | M |
| 2.6 | Language audit: a family-level claim only ever appears next to the suite it came from; the field guide's rigor table is regenerated from a real run | S |

Candidate suites (what is verified vs. still to check):

| Domain | Datasets | Status |
|---|---|---|
| Income / employment / benefits (US census) | folktables ACS tasks: ACSIncome, ACSEmployment, ACSPublicCoverage, ACSMobility, ACSTravelTime, each repeatable over many US states -> a natural multi-dataset suite | tasks verified by search; download of Census PUMS needs network once |
| Lending / credit | German Credit (OpenML), credit-card default and bank marketing (`fairlearn.datasets`, present locally), Adult | fairlearn loaders confirmed present |
| Health | hospital readmission (`fairlearn.datasets.fetch_diabetes_hospital`) | loader confirmed present; more health data to find |
| Criminal justice | COMPAS (race; ProPublica data) | not in fairlearn; manual download, to verify |
| Education | student-performance style data | proposed in the literature; to verify |
| Hiring / HR | no standard public benchmark found | **gap**: ship a documented synthetic suite plus "bring your own" |

Honest limits to keep visible in the product: the literature itself questions whether Adult,
COMPAS and German Credit are good general-purpose fairness benchmarks, and a domain suite is
only as relevant as its resemblance to *your* population.

**Status:** 2.1 registry, 2.3 runner/aggregation/significance/cross-domain comparison, 2.4 console + CLI (with a shortlist that always weighs utility, and absolute-unit tables beside the relative ones), 2.5 repeated CV (`--cv 5x3`) and 2.6 language audit are built and tested. COMPAS is verified on real data. **Not yet verified:** the census, OpenML and fairlearn downloads (blocked in the build sandbox) -- see TESTING.md L1-L8. **Not done:** regenerating the field guide's rigor table from a fresh run (needs `pip install pmlb` and a network); education and extra health suites (no verified source yet).

**Exit:** `ethikit domains run lending` gives per-domain family tables **with fairness columns**
and spreads; the console can show them; no doc claims a cross-domain finding the suite can't support.

## M3 -- Decision support (the "select" half of selection-and-audit)  **[BUILT]**

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

## M5 -- Deep learning  *(HANDOFF Phase 4, scoped down)*

"Tabular" means data laid out like a spreadsheet: one row per person/loan/case, one column per
feature (CSV files, database tables). Everything ethikit evaluates today is tabular. Images, text
and audio are the other kinds. **Scope decision: deep learning on tabular data is in; the LLM
track is extended work** (below).

| # | Work | Size |
|---|---|---|
| 5.1 | Adapter layer: a model is "anything with predict (+ predict_proba)"; PyTorch and Keras wrappers; an sklearn `MLPClassifier` and a small PyTorch MLP join the zoo as a *deep* family | M |
| 5.2 | Captum attributions (integrated gradients) as the DL explainability tool, alongside SHAP | M |
| 5.3 | DP-SGD (Opacus) as an optional "trained with differential privacy" variant of the MLP, so the privacy tool has a real comparison point | M |
| 5.4 | Console: DL models appear in the model list, with their hyperparameters (layers, learning rate, epochs) in the advanced panel | S |

**Extended work (parked, not in the completion definition).** An LLM track only if both hold:
the *tool* is open source **and** the *model's internals are exposed* (open-weight model run
locally, so activations/logits are inspectable and nothing is sent to a third party). Candidates
to evaluate then: Giskard's LLM scan, DeepEval, DeepTeam (check each for features that need a
paid or hosted service, which would disqualify them). Closed API models are out of scope.

**Exit:** a small PyTorch MLP and a differentially-private variant are audited through the same
console with the same principle columns, formulas and model card, and the privacy tool shows a
measurable difference between them.

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

## Testing

Every milestone has its own automated tests and a hand-checkable visual checklist in
[TESTING.md](TESTING.md). "Done" means both pass.

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

## Decisions (settled)

Settled:
1. **Domains:** as many as have credible reference suites (see M2); order of build = verified-available first (census, lending, health), then the manual-download ones, then the documented gaps.
2. **LLM:** extended work only, and only with open-source tools and exposed internals. **Deep learning on tabular data** is in scope.
3. **Python:** standardise on **3.12** (everything installs except Model Card Toolkit, replaced by a JSON export).
4. **Testing:** every milestone ships with automated tests *and* a manual/visual checklist; see [TESTING.md](TESTING.md). A milestone is done only when both pass.
