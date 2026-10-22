"""M6.2: the datasheet is pre-filled from the data profile, and only with what ethikit knows."""
import json

import numpy as np
import pandas as pd
import pytest

from ethikit import datasheet, session


def _prep(cfg):
    return session.prepare({**cfg, "save": False, "tools": ["performance"],
                            "models": [{"name": "logistic_regression"}]})


def test_prefilled_numbers_match_the_profile():
    p = _prep({"data": {"source": "synthetic"}})
    sheet = datasheet.build_datasheet(p.profile, p.config, p.dataset.feature_names)
    a = {i["id"]: i["answer"] for i in sheet["items"]}
    prof = p.profile
    assert str(prof["n_rows"]) in a["composition.count"]
    assert f"{100 * prof['positive_rate']:.1f}%" in a["composition.label"]
    for g in prof["groups"]:
        assert f"{g['group']}: {g['n']} ({100 * g['share']:.1f}%)" in a["composition.subpopulations"]
    assert f"{prof['base_rate_gap']:.2f}" in a["composition.subpopulations"]
    assert f"AUC {prof['proxy']['auc']:.2f}" in a["composition.sensitive"]
    assert "70.0% / 30.0%" in a["composition.splits"] and "seed 0" in a["composition.splits"]
    assert a["uses.risks"].startswith("Found by ethikit's data profile")


def test_your_own_csv_leaves_the_human_questions_open(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=200), "sex": rng.choice(["f", "m"], 200),
                       "y": rng.integers(0, 2, 200)})
    df.loc[:19, "a"] = np.nan
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    p = _prep({"data": {"source": "csv", "path": str(path), "target": "y", "sensitive": "sex",
                        "exclude_sensitive": True}})
    sheet = datasheet.build_datasheet(p.profile, p.config, p.dataset.feature_names)
    a = {i["id"]: i for i in sheet["items"]}
    assert "a 10.0%" in a["composition.missing"]["answer"]
    assert "NOT given to the models" in a["preprocessing.steps"]["answer"]
    for q in ("motivation.purpose", "collection.consent", "distribution.licence", "uses.not_for"):
        assert a[q]["answer"] is None and a[q]["filled_by"] is None
    open_q = datasheet.blank_answers(sheet)
    assert "motivation.purpose" in open_q and "composition.count" not in open_q


def test_your_answers_win_and_are_attributed_to_you():
    p = _prep({"data": {"source": "synthetic"}})
    sheet = datasheet.build_datasheet(p.profile, p.config)
    merged = datasheet.merge_answers(sheet, {"uses.not_for": "credit decisions", "composition.count": ""})
    a = {i["id"]: i for i in merged["items"]}
    assert a["uses.not_for"]["answer"] == "credit decisions" and a["uses.not_for"]["filled_by"] == "you"
    assert a["composition.count"]["filled_by"].startswith("ethikit")       # empty answer ignored
    assert datasheet.completeness(merged)["by_you"] == 1
    with pytest.raises(KeyError):
        datasheet.merge_answers(sheet, {"no.such.question": "x"})
    md, page = datasheet.to_markdown(merged), datasheet.to_html(merged)
    assert "credit decisions" in md and "credit decisions" in page and "Gebru" in md
    json.dumps(merged)


def test_published_facts_follow_the_data_actually_loaded():
    # Adult falls back to synthetic data offline: its published facts must not be applied then
    sheet = datasheet.build_datasheet({"dataset": "synthetic_biased"}, {"data": {"source": "adult"}})
    a = {i["id"]: i["answer"] for i in sheet["items"]}
    assert "Census" not in (a["motivation.creators"] or "") and "ethikit" in a["motivation.creators"]
    sheet = datasheet.build_datasheet({"dataset": "adult_income"}, {"data": {"source": "adult"}})
    by = {i["id"]: i["filled_by"] for i in sheet["items"]}
    assert "archive.ics.uci.edu" in by["motivation.creators"]
