"""Design system (M11) : contrastes calculés, cohérence MASTER.md ↔ theme.py ↔ config.toml, interdits,
ordre des écrans, et chiffres de l'écran Vérifier identiques aux artefacts JSON."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import tomllib

from ui import theme
from ui.components import proof_guarantees, proof_headline, proof_limits, proof_table

ROOT = Path(__file__).resolve().parents[1]
MASTER = (ROOT / "design-system" / "MASTER.md").read_text(encoding="utf-8")
CONFIG = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))["theme"]
FREEZE = json.loads((ROOT / "engine_freeze.json").read_text(encoding="utf-8"))
FINAL = json.loads((ROOT / "logs" / "final_test_result.json").read_text(encoding="utf-8"))
EMOJI = re.compile("[\U0001f300-\U0001faff☀-➿⭐⭕]")


def test_contrast_ratio_reference_values():
    assert theme.contrast_ratio("#000000", "#FFFFFF") == pytest.approx(21.0)
    assert theme.contrast_ratio("#777777", "#FFFFFF") == pytest.approx(4.48, abs=0.01)


@pytest.mark.parametrize("bg", ["background", "surface"])
@pytest.mark.parametrize("fg", theme.TEXT_COLORS)
def test_every_text_color_meets_wcag_aa(fg, bg):
    assert theme.contrast_ratio(theme.COLORS[fg], theme.COLORS[bg]) >= 4.5


@pytest.mark.parametrize("state", list(theme.TINTS))
def test_state_text_on_its_tint_meets_wcag_aa(state):
    assert theme.contrast_ratio(theme.COLORS[state], theme.TINTS[state]) >= 4.5


def test_master_documents_every_token():
    for hex_value in [*theme.COLORS.values(), theme.OBSERVED, theme.B0, theme.B1, theme.NEUTRAL]:
        assert hex_value.lower() in MASTER.lower(), hex_value


def test_native_theme_uses_the_same_tokens():
    c = theme.COLORS
    assert CONFIG["base"] == "light"
    assert CONFIG["primaryColor"] == c["accent"]
    assert CONFIG["backgroundColor"] == c["background"]
    assert CONFIG["secondaryBackgroundColor"] == c["surface"]
    assert CONFIG["textColor"] == c["text"]
    assert CONFIG["borderColor"] == c["border"]
    assert CONFIG["redColor"] == c["critical"] and CONFIG["redBackgroundColor"] == theme.TINTS["critical"]
    assert CONFIG["yellowColor"] == c["warning"] and CONFIG["yellowBackgroundColor"] == theme.TINTS["warning"]
    assert CONFIG["greenColor"] == c["success"] and CONFIG["greenBackgroundColor"] == theme.TINTS["success"]
    assert CONFIG["baseRadius"] == f"{theme.RADIUS_PX}px"


def test_injected_css_is_sober():
    css = theme.css()
    assert f"max-width: {theme.MAX_WIDTH_PX}px" in css and "tabular-nums" in css
    assert theme.FONT_STACK in css
    for forbidden in ("gradient", "box-shadow", "animation", "@import", "http"):
        assert forbidden not in css, forbidden


def test_no_decorative_emoji_in_the_interface():
    for path in [ROOT / "app.py", ROOT / "ui" / "components.py", *sorted((ROOT / "ui" / "pages").glob("*.py"))]:
        assert not EMOJI.search(path.read_text(encoding="utf-8")), path.name


def test_screen_order_is_verify_understand_buy():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    positions = [source.index(f'title="{t}"') for t in ("Vérifier", "Comprendre", "Acheter")]
    assert positions == sorted(positions)
    assert 'title="Vérifier", default=True' in source


# ---------------------------------------------------------------- écran Vérifier = artefacts


def test_proof_table_is_read_from_artifacts_not_typed():
    table = proof_table(FREEZE, FINAL).set_index("Modèle")
    val = FREEZE["métriques_validation_périmètre_principal"]
    test = {"ML (glissant)": FINAL["résultat_moteur_test"], **FINAL["contexte_test"]}
    for key, label in (("ML (glissant)", "ML — moteur gelé"), ("B1 (glissant)", "B1"), ("B0 (glissant)", "B0")):
        assert table.at[label, "MAE validation"] == val[key]["mae"]
        assert table.at[label, "Biais validation"] == val[key]["bias"]
        assert table.at[label, "MAE test final"] == test[key]["mae"]
        assert table.at[label, "Biais test final"] == test[key]["bias"]


def test_headline_guarantees_and_limits_follow_the_artifacts():
    headline = proof_headline(FREEZE, FINAL)
    assert f"{FINAL['résultat_moteur_test']['mae']:.4f}" in headline
    assert f"{FINAL['contexte_test']['B1 (glissant)']['mae']:.4f}" in headline
    assert "134 744" in headline and "," in headline
    assert all(ok for ok, _ in proof_guarantees(FREEZE, FINAL, freeze_intact=True))
    assert not proof_guarantees(FREEZE, FINAL, freeze_intact=False)[0][0]
    broken = FINAL | {"ouvertures_du_test": 2}
    assert not proof_guarantees(FREEZE, broken, True)[2][0]
    limits = " ".join(proof_limits(FREEZE, FINAL, None))
    assert f"{FINAL['résultat_moteur_test']['bias']:+.4f}" in limits


def test_at_most_three_metrics_per_screen(dev):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    counts = {"Vérifier": len(at.metric)}
    for page, name in (("ui/pages/comprendre.py", "Comprendre"), ("ui/pages/acheter.py", "Acheter")):
        at.switch_page(page).run()
        assert not at.exception, [e.value for e in at.exception]
        counts[name] = len(at.metric)
    assert all(n <= 3 for n in counts.values()), counts


def test_night_zero_width_interval_is_disclosed_as_a_limit():
    from stockvisible.uncertainty import load_report

    report = load_report()
    if report is None:
        pytest.skip("rapport de calibration absent")
    limits = proof_limits(FREEZE, FINAL, report)
    assert any("largeur nulle" in text and "0, 1, 2, 3, 4, 5" in text for text in limits)


def test_tab_icon_is_the_galsen_logo_not_streamlit():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "page_icon=str(FAVICON)" in source
    icon = ROOT / "ui" / "assets" / "galsen-favicon.png"
    header = icon.read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n"  # vraie image PNG
    width, height = int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")
    assert (width, height) == (32, 32)

