import json

import numpy as np
import pandas as pd
import pytest

from ethikit import cli, session
from ethikit.data import make_synthetic_raw, prepare_frame
from ethikit.profile import profile_raw
from ethikit.runs import diff_runs, list_runs, load_run, save_run


# ------------------------------------------------------------------ profile
def test_group_sizes_and_base_rates_match_a_hand_computation():
    raw = make_synthetic_raw(n=2000, minority_share=0.08, seed=4)
    p = profile_raw(raw)
    s = pd.Series(raw.sensitive.astype(str)); y = pd.Series(raw.y)
    for g in p["groups"]:
        m = s == g["group"]
        assert g["n"] == int(m.sum()) and g["base_rate"] == round(float(y[m].mean()), 4)
    assert any("under-represented" in w for w in p["warnings"])


def test_proxy_screen_finds_the_injected_proxy_and_ignores_the_sensitive_column():
    strong = profile_raw(make_synthetic_raw(n=3000, proxy_strength=2.5, seed=1))
    weak = profile_raw(make_synthetic_raw(n=3000, proxy_strength=0.0, seed=1))
    assert strong["proxy"]["auc"] > 0.8 and weak["proxy"]["auc"] < 0.6
    assert strong["proxy"]["top_features"][0]["feature"] == "proxy_income"
    assert all(f["feature"] != "group" for f in strong["proxy"]["top_features"])
    assert any("dropping it would not remove it" in w for w in strong["warnings"])


def test_missingness_is_reported_from_the_original_frame():
    r = np.random.default_rng(0); n = 300
    df = pd.DataFrame({"a": r.normal(size=n), "b": r.normal(size=n), "sex": r.choice(["F", "M"], n),
                       "y": r.choice(["no", "yes"], n)})
    df.loc[:149, "b"] = np.nan
    p = profile_raw(prepare_frame(df, "y", "sex"), df)
    assert p["missing"][0] == {"column": "b", "share_missing": 0.5}
    assert any("more than 20% empty" in w for w in p["warnings"])


def test_session_emits_the_profile_before_the_run_and_puts_it_in_the_summary():
    ev = []
    out = session.execute(session.prepare({"data": {"source": "synthetic"}, "models": [{"name": "decision_tree"}],
                                            "tools": ["performance"]}), on_event=ev.append)
    kinds = [e["type"] for e in ev]
    assert kinds[0] == "profile" and kinds.index("profile") < kinds.index("run_start")
    assert out["summary"]["profile"]["groups"]


# ------------------------------------------------------------------ requirements + stability in a run
CFG = {"data": {"source": "synthetic"}, "models": [{"name": n} for n in ["logistic_regression", "decision_tree", "naive_bayes"]],
       "tools": ["performance", "fairness", "robustness"], "requirements": ["demographic_parity_diff<=0.45"]}


def test_run_summary_carries_requirements_and_stability():
    s = session.execute(session.prepare(CFG))["summary"]
    req = s["requirements"]
    assert req["list"] == ["demographic_parity_diff <= 0.45"]
    assert set(req["feasible"]) <= set(s["objective"]) and "naive_bayes" not in req["feasible"]
    assert s["stability"]["available"] and s["stability"]["first"] in req["feasible"]


def test_a_requirement_typo_fails_before_any_training():
    with pytest.raises(ValueError, match="write it like"):
        session.prepare({**CFG, "requirements": ["parity is low"]})


def test_nothing_feasible_is_explained():
    s = session.execute(session.prepare({**CFG, "requirements": ["accuracy>=0.999"]}))["summary"]
    assert s["requirements"]["feasible"] == []
    assert "no model meets every requirement" in s["stability"]["reason"]


# ------------------------------------------------------------------ run history
def test_every_run_is_saved_and_can_be_listed_loaded_and_reproduced():
    s1 = session.execute(session.prepare(CFG))["summary"]
    rid = s1["history_id"]
    assert [r["run_id"] for r in list_runs()] == [rid]
    loaded = load_run(rid)
    assert loaded["config"]["requirements"] == CFG["requirements"]
    s2 = session.execute(session.prepare(loaded["config"]))["summary"]
    d = diff_runs(rid, s2["history_id"])
    assert d["identical"], d


def test_diff_names_config_and_score_changes():
    a = session.execute(session.prepare(CFG))["summary"]["history_id"]
    b = session.execute(session.prepare({**CFG, "split": {"seed": 9}}))["summary"]["history_id"]
    d = diff_runs(a, b)
    assert not d["identical"]
    assert any(c["key"] == "split.seed" for c in d["config_changes"])
    assert d["score_changes"] and d["metric_changes"]


def test_no_save_and_unknown_run():
    session.execute(session.prepare({**CFG, "save": False}))
    assert list_runs() == []
    with pytest.raises(KeyError, match="no saved run"):
        load_run("nope")


def test_uploaded_inputs_are_made_portable_in_saved_configs():
    rid = save_run({"data": {"source": "upload", "upload_id": "abc"},
                    "models": [{"name": "m", "upload_model_id": "x"}]}, {"objective": {}})
    cfg = load_run(rid)["config"]
    assert "set source='csv'" in cfg["data"]["note"] and "set 'file'" in cfg["models"][0]["note"]


# ------------------------------------------------------------------ CLI
def test_cli_require_profile_stability_and_history(tmp_path, capsys):
    rc = cli.main(["run", "--models", "logistic_regression,decision_tree,naive_bayes", "--tools",
                   "performance,fairness", "--require", "demographic_parity_diff<=0.45", "--out", str(tmp_path / "o")])
    assert rc == 0
    out = capsys.readouterr().out
    for needle in ["data profile", "proxy screen", "requirements:", "feasible:", "how stable is the order?", "saved as run"]:
        assert needle in out, needle
    card = (tmp_path / "o" / "model_cards" / "naive_bayes.html").read_text()
    assert "Requirements you set" in card and "does not</b> meet" in card and "What the data says" in card


def test_cli_config_roundtrip_and_export(tmp_path, capsys):
    cfgfile = tmp_path / "c.json"
    assert cli.main(["run", "--models", "decision_tree", "--tools", "performance", "--seed", "3",
                     "--export-config", str(cfgfile)]) == 0
    cfg = json.loads(cfgfile.read_text())
    assert cfg["split"]["seed"] == 3 and list_runs() == []        # export does not run
    assert cli.main(["run", "--config", str(cfgfile), "--out", str(tmp_path / "o")]) == 0
    res = json.loads((tmp_path / "o" / "results.json").read_text())
    assert res["config"]["split"]["seed"] == 3
    assert cli.main(["run", "--config", str(tmp_path / "o" / "results.json"), "--out", str(tmp_path / "o2")]) == 0
    runs = list_runs()
    assert len(runs) == 2 and cli.main(["runs", "diff", runs[0]["run_id"], runs[1]["run_id"]]) == 0
    assert "identical" in capsys.readouterr().out


def test_cli_profile_command(capsys):
    assert cli.main(["profile", "--synthetic", "n=1500", "proxy_strength=2"]) == 0
    out = capsys.readouterr().out
    assert "data profile" in out and "proxy_income" in out


def test_cli_bad_config_file(tmp_path):
    f = tmp_path / "x.json"; f.write_text('{"hello": 1}')
    assert cli.main(["run", "--config", str(f)]) == 2
