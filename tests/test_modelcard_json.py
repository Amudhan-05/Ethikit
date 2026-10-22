"""M6.1: every model card is also exported in Google's Model Card Toolkit JSON schema."""
import json

import pytest

from ethikit import session
from ethikit.report import mct_json, render_model_cards

jsonschema = pytest.importorskip("jsonschema")


@pytest.fixture(scope="module")
def run():
    p = session.prepare({"data": {"source": "synthetic"}, "save": False,
                         "models": [{"name": "logistic_regression"}, {"name": "decision_tree"}],
                         "tools": ["performance", "fairness", "explainability"],
                         "requirements": ["accuracy>=0.99"]})
    return p, session.execute(p)


def _write(run, tmp_path):
    p, out = run
    s = out["summary"]
    render_model_cards(s["dataset"], out["raw"], out["objective"], s["pareto"], out_dir=str(tmp_path),
                       sensitive_name=s["sensitive"], requirements=s.get("requirements"),
                       data_profile=s.get("profile"))
    return {f.stem.replace(".mct", ""): json.loads(f.read_text()) for f in tmp_path.glob("*.mct.json")}


def test_every_card_validates_against_the_vendored_mct_schema(run, tmp_path):
    cards = _write(run, tmp_path)
    assert set(cards) == {"logistic_regression", "decision_tree"}
    for doc in cards.values():
        mct_json.validate(doc)                    # raises on any schema violation
        assert doc["schema_version"] == "0.0.2"


def test_the_json_carries_the_same_numbers_as_the_run(run, tmp_path):
    p, out = run
    doc = _write(run, tmp_path)["decision_tree"]
    raw = out["raw"]
    acc = float(raw[(raw.model == "decision_tree") & (raw.metric == "accuracy")].value.iloc[0])
    m = {x["type"]: x["value"] for x in doc["quantitative_analysis"]["performance_metrics"]}
    assert m["utility/performance/accuracy"] == f"{acc:.4f}"
    cons = doc["considerations"]
    assert any("Fails hard requirement" in t["description"] for t in cons["tradeoffs"])
    assert any("base rates differ" in r["name"] for r in cons["ethical_considerations"])
    assert any(lim["description"].startswith("explainability:") for lim in cons["limitations"])


def test_the_schema_rejects_what_it_should():
    with pytest.raises(jsonschema.ValidationError):
        mct_json.validate({"schema_version": "0.0.2", "not_a_field": 1})
