"""CV in ordinary runs, the `ethikit domains` CLI, and the streaming /api/domain-run endpoint."""
import json
import urllib.request

import pytest

from ethikit import cli, session
from ethikit.domains import get_domains as real_get_domains
from tests.test_domain_runner import toy_domain


# ------------------------------------------------------------------ repeated CV in single runs
def _cv_cfg(cv):
    return {"data": {"source": "synthetic_custom", "synthetic": {"n": 500}}, "split": {"seed": 3, "cv": cv},
            "models": [{"name": "logistic_regression"}, {"name": "decision_tree"}],
            "tools": ["performance", "fairness"]}


def test_cv_run_reports_mean_and_spread_and_is_reproducible():
    a = session.execute(session.prepare(_cv_cfg([3, 2])))["summary"]
    b = session.execute(session.prepare(_cv_cfg([3, 2])))["summary"]
    assert a["cv"] == [3, 2] and a["objective_std"]
    sd = a["objective_std"]["logistic_regression"]["utility"]
    assert sd is not None and sd >= 0
    assert a["objective"] == b["objective"] and a["objective_std"] == b["objective_std"]


def test_without_cv_there_is_no_spread():
    s = session.execute(session.prepare(_cv_cfg(None)))["summary"]
    assert s["cv"] is None and s["objective_std"] is None


def test_cv_graph_events_come_from_the_first_fold_only():
    ev = []
    session.execute(session.prepare(_cv_cfg([3, 1])), on_event=ev.append)
    kinds = [e["type"] for e in ev]
    assert kinds.count("run_start") == 1 and kinds.count("cv_fold") == 3 and kinds.count("run_done") == 1
    assert kinds.count("tool_done") == 2 * 2            # 2 models x 2 tools, one fold's worth


def test_invalid_cv_and_prefit_with_cv_are_rejected():
    with pytest.raises(ValueError, match="at least 2 folds"):
        session.prepare(_cv_cfg([1, 1]))
    cfg = _cv_cfg([3, 1]); cfg["models"] = [{"name": "m", "file": "x.joblib"}]
    with pytest.raises(ValueError, match="can't be cross-validated"):
        session.prepare(cfg)


# ------------------------------------------------------------------ CLI
def test_cli_domains_list_and_info(capsys):
    assert cli.main(["domains", "list"]) == 0
    out = capsys.readouterr().out
    assert "census" in out and "hiring" in out and "gap" in out
    assert cli.main(["domains", "info", "lending"]) == 0
    info = capsys.readouterr().out
    assert "german_credit" in info and "Caveats" in info and "first run on your machine" in info


def test_cli_domains_unknown_id_and_missing_id():
    with pytest.raises(SystemExit):
        cli.main(["domains", "run", "narnia"])
    with pytest.raises(SystemExit):
        cli.main(["domains", "info"])


def test_cli_domains_run_prints_evidence_caveats_and_writes_json(monkeypatch, tmp_path, capsys):
    import ethikit.domains as D
    monkeypatch.setattr(D, "get_domains", lambda *a, **k: {"toy": toy_domain(4)})
    rc = cli.main(["domains", "run", "toy", "--models", "logistic_regression,decision_tree,naive_bayes",
                   "--tools", "performance,fairness", "--no-cache", "--out", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    for needle in ["family evidence", "real units", "leading family", "starting shortlist", "Caveats", "toy caveat"]:
        assert needle in out, needle
    data = json.loads((tmp_path / "domain_toy.json").read_text())
    assert data["shortlist"]["families"] and len(data["datasets"]) == 4


def test_cli_domains_run_on_a_gap_domain_exits_cleanly(capsys):
    assert cli.main(["domains", "run", "hiring"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error:") and "your own" in err.lower() and "Traceback" not in err


# ------------------------------------------------------------------ server
def _stream(server, cfg):
    req = urllib.request.Request(server + "/api/domain-run", data=json.dumps(cfg).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return [json.loads(l) for l in r if l.strip()]


def test_domain_run_streams_dataset_events_then_a_result(server, monkeypatch):
    import ethikit.domains as D
    monkeypatch.setattr(D, "get_domains", lambda *a, **k: {"toy": toy_domain(3, bad=(1,))})
    ev = _stream(server, {"domain": "toy", "models": ["logistic_regression", "decision_tree", "naive_bayes"],
                          "tools": ["performance", "fairness"], "weights": {"fairness": 0.5}})
    kinds = [e["type"] for e in ev]
    assert kinds[0] == "domain_start" and kinds[-1] == "domain_result"
    assert kinds.count("dataset_done") == 2 and kinds.count("dataset_skip") == 1
    res = ev[-1]
    assert res["shortlist"]["weights_used"]["utility"] > 0 and res["skipped"][0]["id"] == "toy1"


def test_domain_run_errors_are_events_and_the_server_survives(server):
    ev = _stream(server, {"domain": "hiring"})
    assert ev[-1]["type"] == "error" and "your own" in ev[-1]["error"].lower()
    ev = _stream(server, {"domain": "narnia"})
    assert ev[-1]["type"] == "error" and "unknown domain" in ev[-1]["error"]
    assert urllib.request.urlopen(server + "/api/meta").status == 200


def test_meta_lists_domains_with_their_strength_and_caveats(server):
    meta = json.loads(urllib.request.urlopen(server + "/api/meta").read())
    by = {d["id"]: d for d in meta["domains"]}
    assert by["census"]["strength"] == "suite" and by["hiring"]["strength"] == "gap"
    assert all(d["caveats"] for d in meta["domains"])
