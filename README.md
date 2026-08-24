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

## Quick start

```bash
pip install -r requirements.txt        # or: pip install -e ".[full]"
python -m examples.run_demo            # runs offline on synthetic data
streamlit run demo/app.py              # interactive UI
```

The demo works with **zero network** (synthetic biased dataset) and upgrades to the real
UCI Adult dataset automatically when online. It works with **just scikit-learn**;
installing `fairlearn`/`shap`/`giskard` unlocks richer evaluators via graceful fallback.

## How it's structured

```
ethikit/
  principles.py          canonical principle taxonomy (UNESCO/MSFT/IBM/Google + provenance)
  core/types.py          Dataset, EvalContext, EvalResult, TaskType
  evaluators/            one file per ethics tool; all share the Evaluator interface
    base.py              Evaluator ABC + registry  <-- add new tools here
    performance.py       utility axis (accuracy / f1 / auc / r2)
    fairness.py          fairlearn -> numpy fallback (demographic parity, equalized odds)
    explainability.py    surrogate fidelity (shap -> permutation-importance fallback)
    robustness.py        perturbation stability (Giskard/ART later)
    privacy.py           train/test-gap leakage proxy (membership inference later)
  data/loaders.py        Adult (+ synthetic fallback) with sensitive attribute
  benchmark/
    models.py            model zoo grouped by family (linear/tree/kernel/…)
    runner.py            train sweep -> run evaluator battery -> tidy records
  scoring/aggregate.py   normalize -> profile -> Pareto -> optional weighted score
  report/report.py       self-contained HTML report
demo/app.py              Streamlit UI
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
    def evaluate(self, ctx):
        return [EvalResult(self.name, "fairness", InspectionTarget.PREDICTIONS,
                           "my_metric", value=0.9, higher_is_better=True)]
register(MyEvaluator())
```

The runner, scoring, and report pick it up automatically. See `HANDOFF.md` for the full
roadmap (ML → deep learning → LLMs).
