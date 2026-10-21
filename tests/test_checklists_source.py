"""Checklist provenance (M6.4): every checklist says which edition it follows, whether it is
verbatim or paraphrased, and when (or whether) it was compared with the original."""
import hashlib
import re

import pytest

from ethikit import checklists

DEON_IDS = ["A.1", "A.2", "A.3", "A.4", "B.1", "B.2", "B.3", "C.1", "C.2", "C.3", "C.4", "C.5",
            "D.1", "D.2", "D.3", "D.4", "D.5", "E.1", "E.2", "E.3", "E.4"]
# sha256 of "id text" lines of deon/assets/checklist.yml @ 8a31cf7 -- changes iff the text does
DEON_FINGERPRINT = "f927d8b6c931ab832631c7cabe5d9106b3d7f8e3e7898e473f01ea3109c6fdec"


@pytest.mark.parametrize("cid", list(checklists.CHECKLISTS))
def test_every_checklist_records_its_provenance(cid):
    c = checklists.get(cid)
    assert c.version and c.published and c.verified_against
    assert c.fidelity in ("verbatim", "paraphrased")
    assert all(i.ref for i in c.items), "every item names where it comes from in the original"
    prov = checklists.as_dict(c)["provenance"]
    assert prov["line"] and prov["fidelity"] == c.fidelity
    # honesty: a date is only claimed when the comparison was actually made
    if c.verified_on is None:
        assert "not machine-checked" in c.verified_against
    else:
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", c.verified_on)


def test_deon_is_the_verbatim_original():
    c = checklists.get("deon")
    assert c.fidelity == "verbatim" and c.original_items == len(c.items) == 21
    assert [i.id for i in c.items] == DEON_IDS
    text = "\n".join(i.id + " " + i.text for i in c.items)
    assert hashlib.sha256(text.encode()).hexdigest() == DEON_FINGERPRINT
    assert any("A.4" in n for n in c.drift_notes), "the drift that was fixed stays on record"


def test_nist_and_altai_refs_point_at_real_parts_of_the_original():
    for i in checklists.get("nist_ai_rmf").items:
        assert re.fullmatch(r"(GOVERN|MAP|MEASURE|MANAGE) \d+\.\d+(, (\w+ )?\d+\.\d+)*", i.ref), i.ref
        fn = {"GV": "GOVERN", "MP": "MAP", "MS": "MEASURE", "MG": "MANAGE"}[i.id.split(".")[0]]
        assert i.ref.startswith(fn)
    for i in checklists.get("altai").items:
        assert i.ref.startswith(f"Req. {i.id.split('.')[0]} -- ")


def test_markdown_export_carries_the_provenance():
    md = checklists.to_markdown("nist_ai_rmf")
    assert "AI RMF 1.0" in md and "not yet verified" in md and "[orig: GOVERN 1.1]" in md
    assert "verbatim" in checklists.to_markdown("deon")


@pytest.mark.network
def test_deon_still_matches_upstream():
    """Live check (needs github.com): re-fetch the pinned upstream file and compare."""
    yaml = pytest.importorskip("yaml")
    import urllib.request
    url = ("https://raw.githubusercontent.com/drivendataorg/deon/"
           "8a31cf751e407ece9d150e2db4fbb644e4fb0499/deon/assets/checklist.yml")
    try:
        src = yaml.safe_load(urllib.request.urlopen(url, timeout=20).read())
    except Exception as exc:
        pytest.skip(f"offline: {exc}")
    upstream = [(ln["line_id"], ln["line"]) for s in src["sections"] for ln in s["lines"]]
    assert upstream == [(i.id, i.text) for i in checklists.get("deon").items]
