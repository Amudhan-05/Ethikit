"""Drives the real console in headless Chromium. Run: pytest -m e2e
(needs `pip install playwright` and a chromium: `playwright install chromium`,
or PLAYWRIGHT_BROWSERS_PATH pointing at an existing install)."""
import glob
import os

import pytest

pytestmark = pytest.mark.e2e
sync_api = pytest.importorskip("playwright.sync_api")


def _chromium():
    for pat in ("/opt/pw-browsers/chromium", "/opt/pw-browsers/chromium-*/chrome-linux/chrome"):
        hits = glob.glob(pat)
        if hits and os.path.exists(hits[0]):
            return hits[0]
    return None


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch(executable_path=_chromium())
        except Exception as exc:                      # no browser available on this machine
            pytest.skip(f"chromium unavailable: {exc}")
        yield b
        b.close()


@pytest.fixture(params=["dark", "light"])
def page(browser, server, request):
    ctx = browser.new_context(viewport={"width": 1400, "height": 1000}, color_scheme=request.param)
    pg = ctx.new_page()
    pg.errors = []
    pg.on("pageerror", lambda e: pg.errors.append(str(e)))
    pg.on("console", lambda m: m.type == "error" and "favicon" not in m.text and pg.errors.append(m.text))
    pg.goto(server + "/")
    pg.wait_for_selector("#models .chip")
    yield pg
    ctx.close()


def _fast_run(pg):
    """Small config so the test is quick: two models, three light tools."""
    for chip in pg.locator("#tools .chip").all():
        name = chip.inner_text().split("\n")[0].split(" ")[0].strip()
        want = name in ("performance", "fairness", "robustness")
        if ("on" in (chip.get_attribute("class") or "")) != want and "off" not in (chip.get_attribute("class") or ""):
            chip.click()
    pg.click("#run")
    pg.wait_for_function("document.querySelector('#run-label').textContent.length > 0", timeout=120000)


def test_theme_toggle_persists_and_returns_to_auto(page, server):
    page.click(".ek-theme button[data-mode=light]")
    assert page.evaluate("document.documentElement.dataset.theme") == "light"
    page.reload()
    assert page.evaluate("document.documentElement.dataset.theme") == "light"
    page.click(".ek-theme button[data-mode=light]")
    assert page.evaluate("document.documentElement.dataset.theme") is None


def test_command_line_follows_clicks(page):
    before = page.inner_text("#cmd-preview")
    page.locator("#models .chip").nth(3).click()
    after = page.inner_text("#cmd-preview")
    assert after != before and after.startswith("ethikit run")      # the » prompt is CSS, not text


def _table(page):
    rows = page.locator("#tab-results table.t tr")
    head = [h.strip() for h in rows.nth(0).locator("th").all_inner_texts()]
    body = [[c.strip() for c in rows.nth(i).locator("td").all_inner_texts()] for i in range(1, rows.count())]
    return head, body


def test_full_run_graph_results_and_reweighting(page):
    _fast_run(page)
    assert page.locator(".g-tool.done").count() >= 3                    # tools lit up in the graph
    head, body = _table(page)
    assert len(body) >= 2 and "weighted" in head
    fi = head.index("fairness") - 1                                   # td index (first col is a td)
    inputs = page.locator("#weights input")
    for i in range(inputs.count()):
        inputs.nth(i).fill("0")
    page.locator("#weights input").nth(0).fill("1")                    # fairness only
    head, body = _table(page)                                          # re-rendered, not re-run
    best = max(body, key=lambda r: float(r[fi + 1]))
    assert body[0][0] == best[0], "top row must be the model with the best fairness score"
    assert not page.errors, page.errors


def test_inspect_shows_formula_and_worked_steps(page):
    _fast_run(page)
    page.click("[data-tab=inspect]")
    assert "=" in page.inner_text("#tab-inspect pre.code")
    assert page.locator("#tab-inspect ol.steps li").count() >= 1
    assert not page.errors, page.errors


def test_checklist_attaches_to_the_model_card(page):
    _fast_run(page)
    page.click("[data-tab=gov]")
    page.locator("#tab-gov input[type=radio][value=yes]").nth(0).check()
    page.click("text=attach to model cards")
    page.wait_for_selector("#tab-gov >> text=✓ attached")
    page.click("[data-tab=card]")
    frame = page.frame_locator("iframe.card")
    frame.locator("text=Governance checklist").first.wait_for(timeout=10000)
    assert frame.locator("text=Deon data-science ethics checklist").count() >= 1


def test_phone_width_has_no_sideways_scroll(browser, server):
    ctx = browser.new_context(viewport={"width": 390, "height": 844})
    pg = ctx.new_page(); pg.goto(server + "/"); pg.wait_for_selector("#models .chip")
    assert pg.evaluate("document.documentElement.scrollWidth") <= 392
    ctx.close()


# ------------------------------------------------------------------ M2: reference suites + cv
def _light_tools(pg):
    for chip in pg.locator("#tools .chip").all():
        name = chip.inner_text().split("\n")[0].split(" ")[0].strip()
        want = name in ("performance", "fairness")
        cls = chip.get_attribute("class") or ""
        if ("on" in cls) != want and "off" not in cls:
            chip.click()


def test_suite_mode_hides_my_data_rows_and_rewrites_the_command(page):
    assert page.locator("#row-domain").is_hidden()
    page.check("input[name=mode][value=suite]")
    assert page.locator("#row-domain").is_visible() and page.locator("#row-suite-caveats").is_visible()
    assert page.locator("#row-modelfile").is_hidden() and page.locator("#data-source").is_hidden()
    assert page.inner_text("#cmd-preview").startswith("ethikit domains run census")
    page.select_option("#domain", "hiring")
    assert "gap" in page.inner_text("#domain-strength")
    assert page.locator("#run").is_disabled()                          # nothing to run: bring your own data
    assert "your own" in page.inner_text("#suite-caveats").lower()
    page.check("input[name=mode][value=mine]")
    assert page.locator("#data-source").is_visible()


def test_suite_run_shows_family_evidence_skips_and_hands_a_shortlist_back(page, monkeypatch):
    import ethikit.domains as D
    from tests.test_domain_runner import toy_domain
    monkeypatch.setattr(D, "get_domains", lambda *a, **k: {"toy": toy_domain(4, bad=(2,))})
    page.goto(page.url)                                               # reload so /api/meta lists the toy domain
    page.wait_for_selector("#models .chip")
    _light_tools(page)
    page.check("input[name=mode][value=suite]")
    page.select_option("#domain", "toy")
    page.click("#run")
    page.wait_for_selector("#tab-suite table.t", timeout=120000)
    text = page.inner_text("#tab-suite")
    assert "family evidence" in text.lower() and "real units" in text.lower() and "caveats" in text.lower()
    assert "toy caveat" in text and "3 dataset(s) ran" in text and "1 skipped" in text
    assert page.locator("#suite-datasets .ds-row.skip").count() == 1
    assert page.locator("#suite-datasets .ds-row.ok").count() == 3
    page.click("#use-shortlist")
    assert page.is_checked("input[name=mode][value=mine]")
    assert page.inner_text("#cmd-preview").startswith("ethikit run")
    assert not page.errors, page.errors


def test_cv_flag_appears_in_the_command_and_the_table_shows_spread(page):
    _light_tools(page)
    page.fill("#cv", "3x1")
    assert "--cv 3x1" in page.inner_text("#cmd-preview")
    page.click("#run")
    page.wait_for_function("document.querySelector('#run-label').textContent.length > 0", timeout=120000)
    assert "±" in page.inner_text("#tab-results table.t")
    assert not page.errors, page.errors


# ------------------------------------------------------------------ M3: decision support
def test_requirement_chip_flows_to_command_results_and_exclusions(page):
    _light_tools(page)
    page.fill("#req-input", "demographic_parity_diff<=0.42")
    page.click("#req-add")
    assert '--require "demographic_parity_diff<=0.42"' in page.inner_text("#cmd-preview")
    page.fill("#req-input", "parity is low")
    page.click("#req-add")
    assert "write it like" in page.inner_text("#req-err")
    page.click("#run")
    page.wait_for_function("document.querySelector('#run-label').textContent.length > 0", timeout=120000)
    txt = page.inner_text("#tab-results").lower()          # headings are CSS-uppercased
    assert "requirements" in txt and "feasible:" in txt and "how stable is the order?" in txt
    assert page.locator("#tab-results .pill.red").count() + page.locator("#tab-results .pill:text('meets')").count() >= 3
    rows = page.locator("#tab-results table.t tr")
    classes = [rows.nth(i).get_attribute("class") or "" for i in range(1, rows.count())]
    assert classes == sorted(classes, key=lambda c: "excluded" in c), "feasible models must come first"
    assert not page.errors, page.errors


def test_data_profile_tab_shows_groups_proxy_and_warnings(page):
    _light_tools(page)
    page.click("#run")
    page.wait_for_function("document.querySelector('#run-label').textContent.length > 0", timeout=120000)
    page.click("[data-tab=data]")
    txt = page.inner_text("#tab-data").lower()             # headings are CSS-uppercased
    assert "before any model" in txt and "proxy screen" in txt and "auc" in txt
    assert "base rates differ" in txt                      # the synthetic data is biased on purpose


def test_history_compare_and_load_config(page):
    _light_tools(page)
    for seed in ("0", "4"):
        page.fill("#seed", seed)
        page.click("#run")
        page.wait_for_function("document.querySelector('#run-label').textContent.length > 0", timeout=120000)
        page.evaluate("document.querySelector('#run-label').textContent=''")
    page.click("[data-tab=history]")
    page.wait_for_selector("#tab-history .hsel")
    boxes = page.locator("#tab-history .hsel")
    assert boxes.count() >= 2
    boxes.nth(0).check(); boxes.nth(1).check()
    page.click("#hdiff")
    page.wait_for_selector("#hdiff-out h3")
    assert "split.seed" in page.inner_text("#hdiff-out")
    page.fill("#seed", "11")
    page.locator("#tab-history .hload").nth(1).click()
    page.wait_for_function("document.querySelector('#seed').value === '0'")


def test_export_then_import_config_roundtrip(page, tmp_path):
    page.locator("#models .chip").nth(3).click()
    page.fill("#req-input", "accuracy>=0.5"); page.click("#req-add")
    page.fill("#cv", "3x2")
    with page.expect_download() as dl:
        page.click("#export-config")
    path = tmp_path / "cfg.json"
    dl.value.save_as(path)
    cmd_before = page.inner_text("#cmd-preview")
    page.reload(); page.wait_for_selector("#models .chip")
    assert page.inner_text("#cmd-preview") != cmd_before
    page.set_input_files("#import-config", str(path))
    page.wait_for_function("document.querySelector('#cv').value === '3x2'")
    assert page.inner_text("#cmd-preview") == cmd_before
