"""Génère les deux PDF du dossier de soumission à partir du template unique.

Chiffres du test final : lus directement dans logs/final_test_result.json (jamais recopiés).
Style : jetons de ui/theme.py (mêmes couleurs, police, espacement que l'application).
Usage : .venv/bin/python submission/build.py
"""

from __future__ import annotations

import html
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from ui import theme

FINAL = json.loads((ROOT / "logs" / "final_test_result.json").read_text(encoding="utf-8"))
ORDER = ("ML (glissant)", "B1 (glissant)", "B0 (glissant)")
VERBATIM_KEYS = ("ouvertures_du_test", "horodatage_début_utc", "horodatage_fin_utc", "moteur_gelé",
                 "résultat_moteur_test", "contexte_test")


def esc(text: str) -> str:
    return html.escape(str(text), quote=False)


def paragraphs(items: list[str]) -> str:
    return "\n".join(f"<p>{esc(t)}</p>" for t in items)


def bullets(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{esc(t)}</li>" for t in items) + "</ul>"


def number(value: float, lang: str, digits: int = 4) -> str:
    text = f"{value:+.{digits}f}" if value < 0 else f"{value:.{digits}f}"
    text = text.replace("-", "−")
    return text.replace(".", ",") if lang == "fr" else text


def results_table(content: dict) -> str:
    lab, lang = content["labels"], content["lang"]
    test = {f"{FINAL['moteur_gelé']} (glissant)": FINAL["résultat_moteur_test"], **FINAL["contexte_test"]}
    val = FINAL["rappel_validation_M8"]
    rows = []
    for key in ORDER:
        cls = ' class="engine"' if key.startswith(FINAL["moteur_gelé"]) else ""
        rows.append(
            f"<tr{cls}><td>{esc(content['models'][key])}</td>"
            f"<td class='num'>{number(val[key]['mae'], lang)}</td>"
            f"<td class='num'>{number(test[key]['mae'], lang)}</td>"
            f"<td class='num'>{number(test[key]['bias'], lang)}</td></tr>"
        )
    head = (f"<tr><th>{esc(lab['table_model'])}</th><th>{esc(lab['table_val_mae'])}</th>"
            f"<th>{esc(lab['table_test_mae'])}</th><th>{esc(lab['table_test_bias'])}</th></tr>")
    return f"<table class='results'>{head}{''.join(rows)}</table>"


def verbatim() -> str:
    """Extrait du fichier, valeurs inchangées ; un objet imbriqué par ligne pour tenir la page."""
    dump = lambda v: json.dumps(v, ensure_ascii=False, separators=(", ", ": "))
    lines = ["{"]
    for i, key in enumerate(VERBATIM_KEYS):
        value = FINAL[key]
        comma = "," if i < len(VERBATIM_KEYS) - 1 else ""
        if key == "contexte_test":
            inner = [f'    {dump(name)}: {dump(stats)}' for name, stats in value.items()]
            lines.append(f'  {dump(key)}: {{\n' + ",\n".join(inner) + f"\n  }}{comma}")
        else:
            lines.append(f"  {dump(key)}: {dump(value)}{comma}")
    lines.append("}")
    return esc("\n".join(lines))


def evidence_table(content: dict) -> str:
    lab = content["labels"]
    head = (f"<tr><th style='width:22%'>{esc(lab['evidence_criterion'])}</th>"
            f"<th>{esc(lab['evidence_proof'])}</th><th style='width:33%'>{esc(lab['evidence_test'])}</th></tr>")
    body = "".join(
        f"<tr><td>{esc(c)}</td><td>{esc(p)}</td><td class='test'>{esc(t)}</td></tr>"
        for c, p, t in content["evidence"]
    )
    return f"<table class='evidence'>{head}{body}</table>"


def table(head: list[str], rows: list[list[str]], mono_cols: tuple[int, ...] = (), cls: str = "") -> str:
    th = "".join(f"<th>{esc(h)}</th>" for h in head)
    body = "".join(
        "<tr>" + "".join(
            f"<td class='mono'>{esc(c)}</td>" if i in mono_cols else f"<td>{esc(c)}</td>" for i, c in enumerate(row)
        ) + "</tr>"
        for row in rows
    )
    return f"<table class='{cls}'><tr>{th}</tr>{body}</table>"


def theme_vars() -> str:
    lines = [f"  --{k.replace('_', '-')}: {v};" for k, v in theme.COLORS.items()]
    lines += [f"  --space-{i}: {px}px;" for i, px in enumerate(theme.SPACING)]
    lines += [f"  --radius: {theme.RADIUS_PX}px;", f"  --font: {theme.FONT_STACK};"]
    return "\n".join(lines)


def render(template: str, content: dict) -> str:
    fragments = {
        "theme_vars": theme_vars(),
        "in_short": bullets(content["cover"]["in_short"]),
        "problem": paragraphs(content["problem"]),
        "data": bullets(content["data"]),
        "method": paragraphs(content["method"]),
        "results_table": results_table(content),
        "verbatim": verbatim(),
        "architecture": paragraphs(content["architecture"]),
        "limits": bullets(content["limits"]),
        "rai": bullets(content["rai"]),
        "evidence_table": evidence_table(content),
        "ai_usage": bullets(content["ai_usage"]),
        "roadmap": bullets(content["roadmap"]),
        "pending": bullets(content["pending"]),
        "screens_table": table([content["labels"]["screen_col"], content["labels"]["screen_desc_col"]],
                               content["screens"], cls="screens"),
        "discipline": bullets(content["discipline"]),
        "architecture_table": table([content["labels"]["fn_col"], content["labels"]["modules_col"],
                                     content["labels"]["role_col"]], content["architecture_table"],
                                    mono_cols=(1,), cls="architecture"),
        "install": esc("\n".join(content["install"])),
        "tests": esc("\n".join(content["tests"])),
        "deploy_table": table([content["labels"]["field_col"], content["labels"]["value_col"]], content["deploy"],
                              mono_cols=(1,), cls="deploy"),
        "licences": bullets(content["licences"]),
    }
    out = re.sub(r"\{\{\{(\w+)\}\}\}", lambda m: fragments[m.group(1)], template)

    def lookup(match: re.Match) -> str:
        value = content
        for part in match.group(1).split("."):
            value = value[part]
        return esc(value)

    out = re.sub(r"\{\{([\w.]+)\}\}", lookup, out)
    assert "{{" not in out and "<script" not in out.lower()
    return out


def main() -> None:
    template = (HERE / "dossier_template.html").read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        for lang, name in (("fr", "FR"), ("en", "EN")):
            content = json.loads((HERE / f"content_{lang}.json").read_text(encoding="utf-8"))
            page = Path(tmp) / f"dossier_{lang}.html"
            page.write_text(render(template, content), encoding="utf-8")
            target = HERE / f"StockVisible_Dossier_{name}.pdf"
            subprocess.run(["node", str(HERE / "print_pdf.js"), str(page), str(target), lang], check=True,
                           cwd=ROOT)
            print(f"écrit : {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
