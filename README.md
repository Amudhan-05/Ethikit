# ethikit

An open-source **orchestrator for evaluating the ethicality of machine-learning models**.
You give it a dataset and a set of models; it runs a battery of open-source ethics
tools, tags every measurement with the ethical principle it addresses, and produces a
per-model profile, a Pareto front, and an HTML report — so a human can choose the model
that best fits their context.

Built around the principle sets published by **UNESCO, Microsoft, IBM, and Google**.

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
so "best model per family, in general" is an aggregate finding, not a one-dataset story.
See `examples/run_phase2_demo.py`.

A fourth thing: the Phase-1 privacy check and the Gaussian-noise robustness check were
both intentionally-labelled proxies, not attacks. Phase 3 replaces/complements them with
real ones: `ethikit.evaluators.privacy` now runs an actual black-box membership-inference
attack (confidence/loss-threshold, Yeom et al. 2018) instead of a train/test-gap heuristic;
`ethikit.evaluators.art_robustness` runs a real adversarial evasion attack (HopSkipJump,
via the Adversarial Robustness Toolbox) instead of random noise; `ethikit.evaluators.drift`
runs Alibi Detect's actual statistical drift test against a manufactured distribution
shift. See `examples/run_phase3_demo.py`.

## Quick start

```bash
pip install -r requirements.txt        # or: pip install -e ".[full,phase2,phase3]"
python -m ethikit serve --open         # the console: live flow graph, inspect, model cards, checklists
python -m ethikit run --trace          # the same run in the terminal (see `ethikit run --help`)
python -m examples.run_demo            # single-dataset demo; runs offline on synthetic data
streamlit run demo/app.py              # legacy Streamlit UI
python -m examples.run_phase2_demo     # generalization: CV, significance, meta-benchmark
python -m examples.run_phase3_demo     # real attacks: membership inference, adversarial, drift
```

The demo works with **zero network** (synthetic biased dataset) and upgrades to the real
UCI Adult dataset automatically when online. It also works with **just scikit-learn** --
every evaluator prefers `fairlearn`/`shap`/`giskard` when installed and falls back to a
dependency-free implementation when they aren't. `aif360` 0.6.1 now installs on Python
3.11-3.13 and runs as its own evaluator (disparate impact / 80% rule, equal opportunity,
average odds, Theil index). `alibi-detect` doesn't build on 3.13 and `model-card-toolkit`
doesn't build on 3.12+, so **Python 3.11 or 3.12** gives you the fullest battery (see
`requirements.txt`).

## The console and the CLI

`python -m ethikit serve` opens a terminal-style web console (standard library only, no extra
install). Every control is a CLI flag, and the console shows the equivalent `ethikit run …` line
as you click. Pressing run streams the sweep live through a flow graph: inputs → each tool,
grouped under the principle it measures → results per principle. After the run you get:

- **results** -- the leaderboard (re-ranks instantly when you change the weights), radar, Pareto front
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
ethikit run --tool-param robustness.sigma=0.25                       # any tool parameter
ethikit list params random_forest | ethikit list tools | ethikit checklist altai
```

The docs site's [console page](https://amudhan-05.github.io/Ethikit/console/) replays a recorded
real run (GitHub Pages can't run Python); refresh it with `python -m ethikit export-site docs`.
Everything follows your system light/dark setting, with a ☀/☾ toggle to override it.

## Docs

**Read them online: https://amudhan-05.github.io/Ethikit/** (GitHub Pages, served from
[`docs/`](docs/); GitHub's file viewer only shows HTML source, so use these links instead).

| Page | What it's for |
|---|---|
| [Console](https://amudhan-05.github.io/Ethikit/console/) | The console, replaying a recorded run -- the same UI `python -m ethikit serve` gives you locally. |
| [Project Proposal](https://amudhan-05.github.io/Ethikit/proposal.html) | The original pitch: the problem, the four-body principle taxonomy, and where the project is headed. |
| [Ethikit Internals](https://amudhan-05.github.io/Ethikit/understanding.html) | The technical companion: every component's mechanism, design rationale, and real-world application angle -- built for learning how and why it's built this way. |
| [Field Guide](https://amudhan-05.github.io/Ethikit/field-guide.html) | A live-demo script (cue cards, timing, what to say) plus a component reference -- built for presenting ethikit to an audience. |
| [Sample Report](https://amudhan-05.github.io/Ethikit/report.html) | A real aggregate report from `run_demo.py`. |

The source files live in [`docs/`](docs/) and also open directly in a browser if you clone the repo.

`docs/report.html` is a point-in-time snapshot of an aggregate report generated by
`run_demo.py`; regenerate it any time with `python -m examples.run_demo` (the fresh copy
lands at repo root as `ethikit_report.html`, gitignored -- copy it into `docs/` if you
want to update the committed snapshot).

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
                          for "best model per family in general" (not one dataset)
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
