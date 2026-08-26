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

## Quick start

```bash
pip install -r requirements.txt        # or: pip install -e ".[full]"
python -m examples.run_demo            # runs offline on synthetic data
streamlit run demo/app.py              # interactive UI
```

The demo works with **zero network** (synthetic biased dataset) and upgrades to the real
UCI Adult dataset automatically when online. It also works with **just scikit-learn** --
every evaluator prefers `fairlearn`/`shap`/`giskard` when installed and falls back to a
dependency-free implementation when they aren't (`aif360` and `model-card-toolkit` are
two examples that don't currently build on Python 3.12; see `requirements.txt`).

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
    robustness.py        perturbation stability (dependency-free)
    privacy.py           train/test-gap leakage proxy (membership inference later)
    giskard_scan.py       unified scan (subgroup performance bias -> fairness, perturbation
                          robustness -> safety); ART / data-leakage scan later
  data/loaders.py        Adult (+ synthetic fallback) with sensitive attribute
  benchmark/
    models.py            model zoo grouped by family (linear/tree/kernel/…)
    runner.py            train sweep -> run evaluator battery -> tidy records
  scoring/aggregate.py   normalize -> profile -> Pareto -> optional weighted score
  report/
    report.py            self-contained aggregate HTML report
    model_card.py         one Model Card per model (schema-compliant HTML;
                          model-card-toolkit sidecar JSON when installable)
demo/app.py              Streamlit UI (Plotly radar + per-model Model Card viewer)
examples/run_demo.py     end-to-end CLI demo
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
