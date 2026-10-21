"""M6.3: the audit bundle has everything a reviewer needs, and re-running it reproduces it."""
import csv
import json
import zipfile

import pytest

from ethikit import session
from ethikit.bundle import verify_bundle, write_bundle
from ethikit.cli import main

CFG = {"data": {"source": "synthetic"}, "save": False,
       "tools": ["performance", "fairness", "membership_inference"],
       "models": [{"name": "logistic_regression"}, {"name": "random_forest", "params": {"max_depth": 4}}],
       "requirements": ["accuracy>=0.7"],
       "checklist": {"checklist": "deon", "answers": {"A.1": {"answer": "yes", "note": "consent forms"}}}}


@pytest.fixture(scope="module")
def summary():
    return session.execute(session.prepare(CFG))["summary"]


def test_the_bundle_contains_every_promised_file(summary, tmp_path):
    root = tmp_path / "b"
    write_bundle(str(root), CFG, summary, datasheet_answers={"uses.not_for": "lending"})
    files = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    for f in ["README.md", "MANIFEST.sha256", "config.json", "results.json", "metrics.csv", "traces.json",
              "report.html", "datasheet.md", "datasheet.html", "datasheet.json", "datasheet_answers.json",
              "environment.json", "checklist/deon_answers.json", "checklist/deon.md",
              "model_cards/logistic_regression.html", "model_cards/random_forest.mct.json"]:
        assert f in files, f
    readme = (root / "README.md").read_text()
    for f in files - {"README.md"}:
        assert f"`{f}`" in readme, f"README does not list {f}"
    assert "ethikit bundle verify" in readme and "accuracy >= 0.7" in readme
    assert "lending" in (root / "datasheet.md").read_text()
    assert "consent forms" in (root / "model_cards" / "random_forest.html").read_text()
    assert ">4<" in (root / "model_cards" / "random_forest.html").read_text().replace("</td><td>", ">")
    rows = list(csv.DictReader(open(root / "metrics.csv")))
    assert len(rows) == len([r for r in summary["records"] if r["target"] != "error"])
    env = json.loads((root / "environment.json").read_text())
    assert env["packages"]["scikit-learn"] and "performance" in env["tools"]


def test_rerunning_the_config_reproduces_every_number(summary, tmp_path):
    root = write_bundle(str(tmp_path / "b"), CFG, summary)
    r = verify_bundle(root)
    assert r["ok"], r
    assert r["compared"] == len([x for x in summary["records"] if x["target"] != "error"])


def test_an_edited_number_is_caught_twice(summary, tmp_path):
    root = tmp_path / "b"
    write_bundle(str(root), CFG, summary)
    rows = (root / "metrics.csv").read_text().splitlines()
    head, first = rows[0], rows[1].split(",")
    first[5] = "0.123456"
    (root / "metrics.csv").write_text("\n".join([head, ",".join(first)] + rows[2:]) + "\n")
    r = verify_bundle(str(root))
    assert not r["ok"]
    assert r["tampered"] == ["metrics.csv"]                       # checksum
    assert r["mismatches"][0]["bundle"] == "0.123456"             # re-run


def test_zip_bundle_and_cli(summary, tmp_path, capsys):
    z = tmp_path / "b.zip"
    write_bundle(str(z), CFG, summary)
    with zipfile.ZipFile(z) as zf:
        assert "ethikit_bundle/metrics.csv" in zf.namelist()
    assert main(["bundle", "verify", str(z), "--no-rerun"]) == 0
    assert "verified" in capsys.readouterr().out
    cfg_path = tmp_path / "c.json"
    cfg_path.write_text(json.dumps(CFG))
    assert main(["run", "--config", str(cfg_path), "--no-save", "--out", str(tmp_path / "o"),
                 "--bundle", str(tmp_path / "cb")]) == 0
    assert main(["bundle", "verify", str(tmp_path / "cb")]) == 0
