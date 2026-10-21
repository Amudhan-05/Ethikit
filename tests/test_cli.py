import json

import pytest

from ethikit import cli


def test_flags_become_the_same_config_the_console_builds():
    a = cli.build_parser().parse_args([
        "run", "--synthetic", "n=500", "minority_share=0.2", "--models", "decision_tree",
        "--model-param", "decision_tree.max_depth=3", "--tool-param", "robustness.sigma=0.25",
        "--weights", "fairness=0.9", "--seed", "4", "--no-scale"])
    cfg = cli.config_from_args(a)
    assert cfg["data"]["source"] == "synthetic_custom" and cfg["data"]["synthetic"]["n"] == "500"
    assert cfg["models"] == [{"name": "decision_tree", "params": {"max_depth": "3"}}]
    assert cfg["tool_params"] == {"robustness": {"sigma": "0.25"}}
    assert cfg["weights"]["fairness"] == 0.9 and cfg["split"] == {"test_size": 0.3, "seed": 4, "scale": False, "cv": None}


def test_malformed_param_flag_is_rejected():
    a = cli.build_parser().parse_args(["run", "--model-param", "bad"])
    with pytest.raises(SystemExit):
        cli.config_from_args(a)


def test_list_tools_and_params(capsys):
    assert cli.main(["list", "tools"]) == 0
    assert "membership_inference" in capsys.readouterr().out
    assert cli.main(["list", "params", "robustness"]) == 0
    assert "sigma = 0.1" in capsys.readouterr().out


def test_missing_csv_is_a_one_line_error_not_a_traceback(capsys):
    assert cli.main(["run", "--data", "nope.csv", "--target", "y"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error:") and "Traceback" not in err


def test_full_run_writes_report_cards_and_results(tmp_path, capsys):
    out = tmp_path / "out"
    rc = cli.main(["run", "--models", "decision_tree", "--tools", "performance,fairness",
                   "--out", str(out)])
    assert rc == 0
    assert (out / "report.html").read_text().startswith("<!doctype html>")
    assert (out / "model_cards" / "decision_tree.html").exists()
    res = json.loads((out / "results.json").read_text())
    assert res["pareto"] == ["decision_tree"] and "traces" in res
    assert "pareto front" in capsys.readouterr().out


def test_checklist_template_and_attach(tmp_path):
    f = tmp_path / "a.json"
    assert cli.main(["checklist", "deon", "--out", str(f)]) == 0
    data = json.loads(f.read_text()); data["answers"]["A.1"]["answer"] = "yes"
    f.write_text(json.dumps(data))
    out = tmp_path / "o"
    assert cli.main(["run", "--models", "decision_tree", "--tools", "performance",
                     "--checklist", str(f), "--out", str(out)]) == 0
    card = (out / "model_cards" / "decision_tree.html").read_text()
    assert "Deon data-science ethics checklist" in card and "A.1" in card
