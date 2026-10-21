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
