# ethikit

An open-source **orchestrator for evaluating the ethicality of machine-learning models**.
You give it a dataset and a set of models; it runs a battery of open-source ethics
tools, tags every measurement with the ethical principle it addresses, and produces a
per-model profile, a Pareto front, and an HTML report — so a human can choose the model
that best fits their context.

Built around the principle sets published by **UNESCO, Microsoft, IBM, and Google**.

**Where this is going:** [ROADMAP.md](ROADMAP.md) -- milestones M1-M7 to completion. **How to check it works:** [TESTING.md](TESTING.md) -- automated commands and a visual checklist for every milestone, plus a baseline you can run today.

## What it is -- and isn't

ethikit is a **model-selection and audit tool built on ethical principles**. It is *not* a
model ranker or an ethics scorer. You bring a dataset and candidate models; it runs
open-source tools that each measure one principle, shows every candidate's full profile and
the trade-offs between them, and leaves an auditable trail (formulas, intermediate numbers,
model cards, governance checklists) so a human can choose -- and justify the choice.

## Why it exists

"The most ethical model" is not a single number. Some fairness criteria are
mathematically incompatible (Kleinberg / Chouldechova impossibility results), so
collapsing everything into one score hides the real trade-offs. ethikit instead exposes
a **multi-dimensional profile** and only produces a single ranking once *you* supply
explicit weights — making the value judgement visible rather than baked in.

Two things that follow from that: (1) picking a *model* isn't the only choice a human
makes here -- picking a *tool* is too, once more than one evaluator speaks to the same
principle (`ethikit.coverage.principle_to_tools()` is the reverse of every evaluator's
own `principle_ids`, so you can go from "I care about fairness" to "here's what's
registered for it" in either direction). (2) no score is a black box: every `EvalResult`
carries a `mechanism` string explaining, in this run's actual terms (sample sizes,
backend used, groups compared), what the tool did internally to produce that number --
see any Model Card's Quantitative Analysis table, or a tool's static `Evaluator.mechanism`
via `ethikit.coverage.tool_info()`.

A third thing: one split is one anecdote. `run_demo.py`'s single train/test split can make
a model look best just because it got a lucky test slice -- `ethikit.benchmark.cv` reruns
the same battery across repeated stratified folds so every score becomes a *distribution*,
and `ethikit.scoring.significance` runs a Friedman test (+ Nemenyi post-hoc, + critical-
difference diagram) so "model A beats model B" is a hypothesis test, not eyeballing two
means. `ethikit.benchmark.meta` runs the same idea across *multiple datasets* (via PMLB),
so a *family-level tendency* is an aggregate finding, not a one-dataset story. Caveat: the
default suite is a generic PMLB sample with no sensitive attribute (so fairness does not run
in it) and is not organised by domain; treat it as evidence about utility / robustness /
privacy tendencies, and run per-domain suites yourself before claiming more.
See `examples/run_phase2_demo.py`.

A fourth thing: the Phase-1 privacy check and the Gaussian-noise robustness check were
both intentionally-labelled proxies, not attacks. Phase 3 replaces/complements them with
real ones: `ethikit.evaluators.privacy` now runs an actual black-box membership-inference
attack (confidence/loss-threshold, Yeom et al. 2018) instead of a train/test-gap heuristic;
`ethikit.evaluators.art_robustness` runs a real adversarial evasion attack (HopSkipJump,
via the Adversarial Robustness Toolbox) instead of random noise; `ethikit.evaluators.drift`
runs Alibi Detect's actual statistical drift test against a manufactured distribution
shift. See `examples/run_phase3_demo.py`.

## Run it locally

ethikit runs entirely on your machine: no accounts, no hosting, nothing leaves your computer
(only the optional OpenML datasets, Adult and German Credit, need the network).

**1. Set up** (Python 3.12 is the project standard; 3.13 runs but the drift tool can't install there)

```bash
git clone https://github.com/AmudhanS-05/Ethikit && cd Ethikit
python3.12 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                            # everything, incl. optional tools
# minimal alternative: pip install -r requirements-core.txt # scikit-learn only; tools needing
#                                                          # missing libraries are skipped (and say how to enable them)
```

**2. Start the console** and leave it running

```bash
python -m ethikit serve --open       # http://127.0.0.1:8765  (--port 9000 to change)
```

The console is at `/`; the project docs (proposal, internals, field guide, sample report) are
served by the same command at `/docs/`, so the whole thing works offline.

**3. Or use the terminal**

```bash
python -m ethikit run --trace                         # synthetic data, every model + tool
python -m ethikit run --data loans.csv --target approved --sensitive sex
python -m ethikit list tools | models | datasets | params <name> | checklists
python -m ethikit checklist deon --out answers.json   # governance checklist template
```

Outputs go to `./ethikit_out/` (`report.html`, `model_cards/`, `results.json`); that folder is
disposable. Everything is also pip-installable: `pip install -e .` adds an `ethikit` command.

**Other entry points:** `python -m examples.run_demo` (original single-split demo),
`run_phase2_demo` (cross-validation + significance; the meta-benchmark part needs
`pip install pmlb`), `run_phase3_demo` (membership inference, adversarial attack, drift),
`streamlit run demo/app.py` (older Streamlit UI).

**4. Check it works**

```bash
pip install -r requirements-dev.txt
python -m pytest -m "not e2e"     # ~7 s
python -m pytest -m e2e           # drives the console in headless Chromium (playwright install chromium first)
```

[TESTING.md](TESTING.md) lists what every test guarantees and a visual checklist to verify by eye.

**Troubleshooting**
- *A tool shows "not installed" in the console:* `pip install` its package (see `requirements.txt`); the rest still run.
- *`alibi-detect` fails to build:* use Python 3.11/3.12.
- *Port in use:* `python -m ethikit serve --port 9000`.
- *Loading `--model-file`:* only load files you trust; they are unpickled.

## Choosing, not ranking: requirements, stability, profile, history

```bash
ethikit run --require "demographic_parity_diff<=0.1" --require "membership_inference_advantage<0.2"
ethikit profile --data loans.csv --target approved --sensitive sex   # the data alone, before any model
ethikit runs list                                                    # every run is saved in ./ethikit_runs
ethikit runs diff <run_a> <run_b>                                    # what changed, config and scores
ethikit run --config <run_id | config.json | results.json>           # reproduce a run exactly
ethikit run ... --export-config my_audit.json                        # save a config without running
```

- **Requirements** are hard limits on raw metric values. Models that fail are listed with the
  requirement and the margin, never hidden; an unmeasured metric counts as a failure.
- **Choice stability** says how far your weights would have to move before a different model
  comes first ("decision_tree stays first while fairness's share is 0.00-0.95; above that
  random_forest takes over").
- **Data profile** runs before any model: group sizes, base rates per group, a proxy screen
  (can the other features recover the sensitive attribute?), and missingness.
- **History**: every run is saved locally (`ETHIKIT_RUNS` to move it); diff two runs, or reload
  one's config in the console's *history* tab. *export config* / *import config* do the same by file.

## Tools (14) and the what-if

Every principle a tool can measure has at least two independent tools, and every tool says what
it does **not** measure (shown in *inspect* and on the model card). `ethikit list tools` lists them.

```bash
ethikit whatif --models logistic_regression --method exponentiated_gradient --constraint demographic_parity
ethikit explain --models decision_tree --row 4        # why this one row got its score
```

The **what-if** trains a fairness-constrained version of a model (fairlearn) and shows the cost
before and after. It is a counterfactual: it never changes any audit score.

## Deep learning on tabular data (optional)

"Tabular" = spreadsheet-shaped data: one row per case, one column per feature. The **deep**
family always has scikit-learn's `mlp`. Install PyTorch to add two more (CPU wheels are small):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install captum opacus
python -m ethikit run --family deep --tools performance,membership_inference,captum_ig
```

- `torch_mlp` -- a small PyTorch network that speaks scikit-learn, so every tool, the console
  and the model card treat it like any other model (layers, learning rate, epochs are tunable).
- `torch_mlp_dp` -- the same network trained with **differential privacy** (DP-SGD, Opacus).
  Its model card states the privacy budget actually spent (epsilon, delta).
- `captum_ig` -- integrated gradients read from the network itself, with a completeness check
  (the attributions must add up to the change in the prediction).

On small, noisy data a large plain network memorises its training set; the DP version does not.
One measured example (TESTING.md, M5): membership-inference advantage **0.41 → 0.05** at
epsilon = 1, with test accuracy 0.62 → 0.67. DP can also *widen* a fairness gap (it costs
accuracy most on small groups), so look at the fairness column too -- that is the trade-off
the audit is there to show.

## Reference suites: choosing a starting model before you have data

No data or trained model of your own yet? A **reference suite** runs the same ethics battery over
several public datasets from one domain and reports which *kinds* of model tend to do well there.
It is a prior, never a verdict: re-test the shortlist on your own data as soon as you have it.

```bash
python -m ethikit domains list                       # census, lending, health, justice, hiring (gap)
python -m ethikit domains info lending               # datasets, sources, sensitive attributes, caveats
python -m ethikit domains run census --max-datasets 3
python -m ethikit domains run justice --cv 3x2       # repeated CV: every number gets a spread
```

or in the console: `--mode reference suite`. You get family-level tables (relative 0-1 scores *and*
real units such as accuracy and parity gap), whether the leading family also leads on each dataset,
a Friedman test when there is enough data, and a starting shortlist ("use this shortlist on my
data"). Suites say how thin they are: `gap` (hiring: bring your own data), `single`, `small`, `suite`.
Datasets download once and are cached in `~/.cache/ethikit`. The census suite needs
`pip install folktables`. See [TESTING.md](TESTING.md) (live checks L1-L8) for which loaders have
been verified against real data.

## The console and the CLI

`python -m ethikit serve` opens a terminal-style web console (standard library only, no extra
install). Every control is a CLI flag, and the console shows the equivalent `ethikit run …` line
as you click. Pressing run streams the sweep live through a flow graph: inputs → each tool,
grouped under the principle it measures → results per principle. After the run you get:

- **results** -- the comparison table (re-orders instantly when you change the weights), radar, Pareto front
- **inspect** -- for any model × tool: the formula, the same formula worked through with this
  run's numbers, and the intermediate artefacts (per-group tables, the attacker's
  member/non-member histograms, the surrogate tree's rules, per-feature drift p-values, ...)
- **model-card** -- the full card, now including all of the above and any governance checklist
- **governance** -- condensed Deon, EU ALTAI, Datasheets for Datasets and NIST AI RMF checklists
  for the principles no tool can measure (accountability, human oversight, well-being)

Bring your own data and models:

```bash
ethikit run --data loans.csv --target approved --sensitive sex      # any CSV
ethikit run --synthetic n=2000 minority_share=0.15 bias=1.5          # tunable synthetic data
ethikit run --model-param random_forest.max_depth=5                  # any hyperparameter
ethikit run --estimator xt=sklearn.ensemble.ExtraTreesClassifier     # any estimator class
ethikit run --model-file mine=model.joblib --no-scale ...            # your own trained model, as-is
ethikit run --model-file net=model.pt ...                            # a PyTorch nn.Module (or .keras/.h5)
ethikit run --tool-param robustness.sigma=0.25                       # any tool parameter
ethikit run --cv 5x3                                                 # repeated k-fold: every score gets a spread
ethikit list params random_forest | ethikit list tools | ethikit checklist altai
```

Everything follows your system light/dark setting, with a ☀/☾ toggle to override it.

## Docs

All local. With the console running, open **http://127.0.0.1:8765/docs/**; or open the files in
[`docs/`](docs/) directly in a browser.

| Page | What it's for |
|---|---|
| [Proposal](docs/proposal.html) | The pitch: the problem, the four-body principle taxonomy, and where the project is headed. |
| [Ethikit Internals](docs/understanding.html) | The technical companion: every component's mechanism and design rationale. |
| [Field Guide](docs/field-guide.html) | A timed live-demo script plus a component reference, for presenting ethikit. |
| [Sample Report](docs/report.html) | A real aggregate report from `run_demo.py`. |
| [Console replay](docs/console/index.html) | A static replay of a recorded run, for when you can't run Python. The live console is `python -m ethikit serve`. |

`docs/report.html` is a snapshot generated by `run_demo.py`; refresh the console replay with
`python -m ethikit export-site docs`.

**Teaching material:** `learn/` holds a personal learning workspace. Only general material
(glossary, resources, reference sheets, shared components) is tracked in git; lessons and
anything specific to the learner stay local (see `.gitignore`).

## How it's structured

```
ethikit/
  principles.py          canonical principle taxonomy (UNESCO/MSFT/IBM/Google + provenance)
  coverage.py            reverse map: principle -> tools (+ each tool's static mechanism)
  core/types.py          Dataset, EvalContext, EvalResult (value + mechanism + details), TaskType
  evaluators/            one file per ethics tool; all share the Evaluator interface
    base.py              Evaluator ABC + registry  <-- add new tools here
    performance.py       utility axis (accuracy / f1 / auc / r2)
    fairness.py          fairlearn -> numpy fallback (demographic parity, equalized odds)
    explainability.py    surrogate fidelity + shap attributions -> permutation-importance fallback
    robustness.py        perturbation stability under random Gaussian noise (dependency-free)
    privacy.py           REAL membership-inference attack (confidence/loss-threshold,
                          cross-validated attack classifier) -- no proxy anymore
    giskard_scan.py       unified scan (subgroup performance bias -> fairness, perturbation
                          robustness -> safety); data-leakage scan not wired in
    art_robustness.py     Phase 3: real black-box adversarial attack (HopSkipJump via ART) --
                          worst-case robustness, complementing robustness.py's random-noise case
    drift.py               Phase 3: Alibi Detect's statistical drift test against a manufactured
                          covariate shift + the model's own performance drop under it
  data/loaders.py        Adult (+ synthetic fallback), RawData + split_dataset() so a
                          single split and repeated CV scale/split identically
  benchmark/
    models.py            classifier + regressor zoo grouped by family (linear/tree/kernel/…)
    runner.py            train sweep -> run evaluator battery -> tidy records
    cv.py                 Phase 2: repeated stratified K-fold -> a score DISTRIBUTION
                          per model instead of one split's point estimate
    meta.py                Phase 2: the same sweep across many PMLB datasets, aggregated,
                          for family-level tendencies across datasets (not one dataset; not per-domain)
  scoring/
    aggregate.py          normalize -> profile -> Pareto -> optional weighted score
    significance.py        Phase 2: Friedman + Nemenyi + critical-difference diagram,
                          so "model A beats model B" is backed by a hypothesis test
  report/
    report.py            self-contained aggregate HTML report
    model_card.py         one Model Card per model (schema-compliant HTML;
                          model-card-toolkit sidecar JSON when installable)
demo/app.py              Streamlit UI (Plotly radar + per-model Model Card viewer)
examples/
  run_demo.py            single-dataset, single-split end-to-end CLI demo
  run_phase2_demo.py      CV + significance + meta-benchmark + regression fairness
  run_phase3_demo.py      membership inference + adversarial robustness + drift detection
```

## Extending it (the whole point)

Adding a new ethics tool is one file:

```python
from ethikit.evaluators.base import Evaluator, register
from ethikit.core.types import EvalResult
from ethikit.principles import InspectionTarget

class MyEvaluator(Evaluator):
    name = "my_tool"
    principle_ids = ["fairness"]
    # static, run-independent: how this tool decides, in general -- shows up in
    # ethikit.coverage's reverse principle -> tool lookup
    mechanism = "Explain in one paragraph what this tool actually inspects and how."
    def evaluate(self, ctx):
        return [EvalResult(
            self.name, "fairness", InspectionTarget.PREDICTIONS, "my_metric",
            value=0.9, higher_is_better=True,
            # dynamic: THIS run's actual numbers -- shows up in every Model Card
            mechanism=f"Computed my_metric over {len(ctx.dataset.y_test)} test rows.",
        )]
register(MyEvaluator())
```

The runner, scoring, and report pick it up automatically -- including the coverage
lookup and the Model Card's mechanism column, no extra wiring needed. See `HANDOFF.md`
for the full roadmap (ML → deep learning → LLMs).
