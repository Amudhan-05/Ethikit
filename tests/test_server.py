import http.client
import json
import urllib.request

import pytest


def get(url):
    with urllib.request.urlopen(url) as r:
        return r.status, r.headers.get_content_type(), r.read()


def post(url, data, ctype="application/json"):
    req = urllib.request.Request(url, data=data if isinstance(data, bytes) else json.dumps(data).encode(),
                                 method="POST", headers={"Content-Type": ctype})
    return urllib.request.urlopen(req)


def run_stream(server, cfg):
    events = []
    with post(server + "/api/run", cfg) as r:
        for line in r:
            if line.strip():
                events.append(json.loads(line))
    return events


SMALL = {"data": {"source": "synthetic"}, "models": [{"name": "decision_tree"}],
         "tools": ["performance", "fairness"]}


def test_console_and_meta_are_served(server):
    status, ctype, body = get(server + "/")
    assert status == 200 and ctype == "text/html" and b"Ethikit Console" in body
    meta = json.loads(get(server + "/api/meta")[2])
    assert meta["live"] is True and meta["tools"] and meta["checklists"]


def test_static_assets_have_correct_types(server):
    assert get(server + "/console.js")[1] in ("text/javascript", "application/javascript")
    assert get(server + "/theme.css")[1] == "text/css"


def test_docs_are_served_locally_and_nothing_outside_them(server):
    assert get(server + "/docs/")[0] == 200
    assert get(server + "/docs/proposal.html")[0] == 200
    host, port = server.replace("http://", "").split(":")
    for bad in ["/docs/../ethikit/server.py", "/docs/%2e%2e/README.md", "/../README.md", "/%2e%2e/README.md"]:
        c = http.client.HTTPConnection(host, int(port)); c.request("GET", bad)
        assert c.getresponse().status == 404, bad


def test_run_streams_events_then_a_result(server):
    ev = run_stream(server, SMALL)
    kinds = [e["type"] for e in ev]
    assert kinds[0] == "prepare" and "run_start" in kinds and kinds[-1] == "result"
    res = ev[-1]
    assert res["pareto"] == ["decision_tree"] and res["run_id"]
    assert res["traces"]["decision_tree"]["fairness"]["trace"]["steps"]


def test_bad_config_yields_an_error_event_not_a_crash(server):
    ev = run_stream(server, {"data": {"source": "csv", "path": "x"}, "models": []})
    assert ev[-1]["type"] == "error" and "target" in ev[-1]["error"]
    assert get(server + "/api/meta")[0] == 200                       # server still alive


def test_upload_csv_then_run_on_it(server):
    import numpy as np
    r = np.random.default_rng(0); n = 300
    rows = ["age,job,sex,approved"] + [
        f"{r.integers(18, 70)},{r.choice(['a', 'b'])},{r.choice(['F', 'M'])},{r.choice(['yes', 'no'])}" for _ in range(n)]
    with post(server + "/api/upload?name=loans.csv", "\n".join(rows).encode(), "text/csv") as resp:
        info = json.loads(resp.read())
    assert info["n_rows"] == n and info["suggest"] == {"target": "approved", "sensitive": "sex"}
    ev = run_stream(server, {"data": {"source": "upload", "upload_id": info["upload_id"], "name": "loans",
                                      "target": "approved", "sensitive": "sex"},
                             "models": [{"name": "logistic_regression"}], "tools": ["performance", "fairness"]})
    assert ev[-1]["type"] == "result" and ev[-1]["dataset"] == "loans"


def test_card_report_and_checklist_attach(server):
    res = run_stream(server, SMALL)[-1]
    rid = res["run_id"]
    card = get(f"{server}/api/card/{rid}/decision_tree")[2].decode()
    assert "Model Card: decision_tree" in card and "No governance checklist attached" in card
    answers = {"checklist": "deon", "answers": {"A.1": {"answer": "yes", "note": "documented"}}}
    with post(f"{server}/api/checklist/{rid}", answers) as r:
        assert json.loads(r.read())["ok"] is True
    card2 = get(f"{server}/api/card/{rid}/decision_tree")[2].decode()
    assert "Deon data-science ethics checklist" in card2 and "documented" in card2
    assert b"Ethical AI evaluation report" in get(f"{server}/api/report/{rid}")[2]


def test_unknown_checklist_is_rejected(server):
    rid = run_stream(server, SMALL)[-1]["run_id"]
    with pytest.raises(urllib.error.HTTPError) as e:
        post(f"{server}/api/checklist/{rid}", {"checklist": "made_up", "answers": {}})
    assert e.value.code == 400


def test_result_keeps_the_server_run_id_and_carries_m3_fields(server):
    cfg = {**SMALL, "models": [{"name": "decision_tree"}, {"name": "logistic_regression"}],
           "requirements": ["demographic_parity_diff<=0.5"]}
    ev = run_stream(server, cfg)
    assert ev[0]["type"] == "prepare" and ev[1]["type"] == "profile"
    res = ev[-1]
    assert res["run_id"] == ev[0]["run_id"]                 # not clobbered by the saved-run id
    assert res["history_id"] and res["requirements"]["feasible"] and res["stability"]["available"]
    card = get(f"{server}/api/card/{res['run_id']}/decision_tree")[2].decode()
    assert "Requirements you set" in card and "What the data says" in card


def test_history_endpoints(server):
    a = run_stream(server, SMALL)[-1]["history_id"]
    b = run_stream(server, {**SMALL, "split": {"seed": 5}})[-1]["history_id"]
    runs = json.loads(get(server + "/api/runs")[2])["runs"]
    assert {a, b} <= {r["run_id"] for r in runs}
    assert json.loads(get(f"{server}/api/runs/{a}")[2])["config"]["models"] == SMALL["models"]
    d = json.loads(get(f"{server}/api/runs-diff?a={a}&b={b}")[2])
    assert d["config_changes"]


def test_whatif_and_explain_endpoints(server):
    pytest.importorskip("fairlearn")
    res = run_stream(server, {**SMALL, "models": [{"name": "logistic_regression"}]})[-1]
    with post(server + "/api/whatif", {"run_id": res["run_id"], "model": "logistic_regression",
                                       "method": "threshold_optimizer"}) as r:
        w = json.loads(r.read())
    assert w["label"].startswith("counterfactual") and w["after"]["demographic_parity_diff"] < w["before"]["demographic_parity_diff"]
    e = json.loads(get(f"{server}/api/explain/{res['run_id']}/logistic_regression?row=3")[2])
    assert e["row"] == 3 and e["contributions"]
