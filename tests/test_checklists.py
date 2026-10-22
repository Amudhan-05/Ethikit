import json

import pytest

from ethikit import checklists as C
from ethikit import principles as P


def test_every_item_maps_to_a_real_principle():
    ids = {p.id for p in P.PRINCIPLES}
    for c in C.CHECKLISTS.values():
        assert c.items and c.url.startswith("https://")
        for it in c.items:
            assert set(it.principles) <= ids, (c.id, it.id)


def test_item_ids_are_unique_within_a_checklist():
    for c in C.CHECKLISTS.values():
        ids = [i.id for i in c.items]
        assert len(ids) == len(set(ids)), c.id


def test_scoring_yes_partial_no_and_na():
    c = C.get("deon")
    item = c.items[0]
    p = item.principles[0]
    base = C.principle_summary("deon", {})[p]
    assert base["score"] == 0.0 and base["answered"] == 0           # unanswered = not evidenced
    yes = C.principle_summary("deon", {item.id: {"answer": "yes"}})[p]
    assert yes["answered"] == 1 and yes["score"] > base["score"]
    part = C.principle_summary("deon", {item.id: {"answer": "partial"}})[p]["score"]
    assert base["score"] < part < yes["score"]
    na = C.principle_summary("deon", {item.id: {"answer": "n/a"}})[p]
    assert na["applicable"] == base["applicable"] - 1


def test_scores_are_between_zero_and_one():
    answers = {i.id: {"answer": "yes"} for i in C.get("altai").items}
    for row in C.principle_summary("altai", answers).values():
        assert 0 <= row["score"] <= 1


def test_answers_file_roundtrip_and_validation(tmp_path):
    f = tmp_path / "a.json"
    data = C.blank_answers("nist_ai_rmf")
    f.write_text(json.dumps(data))
    assert C.load_answers(str(f))["checklist"] == "nist_ai_rmf"     # unanswered (None) is allowed
    data["answers"]["GV.1"]["answer"] = "maybe"
    f.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="answer must be one of"):
        C.load_answers(str(f))


def test_markdown_export_marks_answers():
    md = C.to_markdown("deon", {"A.1": {"answer": "yes", "note": "see wiki"}})
    assert "- [x] **A.1**" in md and "see wiki" in md
