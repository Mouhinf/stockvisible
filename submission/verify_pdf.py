"""Vérifie les deux PDF du dossier : pages, taille, texte, tableaux, sources, fidélité FR/EN.

Usage : .venv/bin/python submission/verify_pdf.py  (code de sortie 1 au moindre écart)
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCES = ["README.md", "logs/final_test_result.json", "docs/judging-evidence.md", "AI_USAGE.md"]
# Nombres qui ne sont pas des affirmations : date du dossier, numéros de milestones du calendrier de
# soumission (M18, M19), pagination, numéros de section.
NOT_CLAIMS = {"27", "2026", "18", "19", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}
CAUTION = {
    "fr": ["test de reconstruction / fidélité", "pas une preuve de récupération de la demande réelle",
           "pas la demande perdue réelle"],
    "en": ["reconstruction / fidelity test", "not proof of recovering real demand", "not the real lost demand"],
}
# Toute mention d'une « preuve de récupération » doit être niée ; aucune affirmation plus forte.
NEGATED = {"fr": ("preuve de récupération", "pas une "), "en": ("proof of recovering", "not ")}
FORBIDDEN = ["prouve la demande", "prouve que la demande", "proves the lost demand", "proves that demand",
             "demande réelle récupérée", "recovered real demand"]


def text(pdf: Path, page: int | None = None) -> str:
    args = ["pdftotext", "-enc", "UTF-8"]
    if page:
        args += ["-f", str(page), "-l", str(page)]
    return subprocess.run([*args, str(pdf), "-"], capture_output=True, text=True, check=True).stdout


def pages(pdf: Path) -> int:
    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    return int(re.search(r"Pages:\s+(\d+)", info).group(1))


def numbers(raw: str) -> list[str]:
    t = raw.replace("−", "-").replace(" ", " ").replace(" ", " ")
    t = t.replace("SHA256", "SHA-256")  # pdftotext recolle « SHA- » + « 256 » coupé en fin de ligne
    t = re.sub(r"\b(\d{1,2}):00\b", r"\1", t)  # 22:00 (EN) = 22 h (FR)
    t = re.sub(r"(?<=\d) (?=\d{3}\b)", "", t)  # 134 744 -> 134744
    t = re.sub(r"(?<=\d),(?=\d)", ".", t)  # 0,0492 -> 0.0492
    t = re.sub(r"\bPage \d+ / \d+\b", " ", t)
    found = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", t)
    return [n.lstrip("-").rstrip(".") for n in found]


def source_blob() -> str:
    blob = " ".join((ROOT / s).read_text(encoding="utf-8") for s in SOURCES)
    blob = blob.replace("−", "-").replace(" ", " ").replace(" ", " ")
    blob = re.sub(r"(?<=\d) (?=\d{3}\b)", "", blob)
    return re.sub(r"(?<=\d),(?=\d)", ".", blob)


def check(lang: str, name: str, blob: str) -> tuple[list[str], list[str], Counter]:
    pdf = HERE / f"StockVisible_Dossier_{name}.pdf"
    problems, report = [], []
    n = pages(pdf)
    size = pdf.stat().st_size
    full = text(pdf)
    raw = pdf.read_bytes()
    images = subprocess.run(["pdfimages", "-list", str(pdf)], capture_output=True, text=True, check=True).stdout
    n_images = max(0, len(images.strip().splitlines()) - 2)
    report.append(f"{pdf.name}: {n} pages, {size / 1024:.0f} Ko, {len(full)} caractères de texte, {n_images} image(s)")
    if not 3 <= n <= 4:
        problems.append(f"{name}: {n} pages hors cible 3-4")
    if size >= 5 * 1024 * 1024:
        problems.append(f"{name}: taille {size} >= 5 Mo")
    if len(full) < 3000 or n_images:
        problems.append(f"{name}: texte non sélectionnable ou images présentes")
    for marker in (b"/JavaScript", b"/JS ", b"/Annots", b"/AcroForm"):
        if marker in raw:
            problems.append(f"{name}: élément interactif {marker!r}")

    # Tableaux et bloc verbatim : premier et dernier repère sur la même page.
    per_page = [text(pdf, p) for p in range(1, n + 1)]
    spans = {
        "tableau des résultats": ("0.0456" if lang == "en" else "0,0456", "0.0475" if lang == "en" else "0,0475"),
        "tableau critère → preuve": ("test_problem_banner", "test_real_report_uses_validation"),
        "extrait verbatim": ('"ouvertures_du_test"', '"B1 (glissant)"'),
    }
    for label, (first, last) in spans.items():
        where = [i for i, page in enumerate(per_page, 1) if first in page]
        where_last = [i for i, page in enumerate(per_page, 1) if last in page]
        if not where or not where_last or where[0] != where_last[-1]:
            problems.append(f"{name}: {label} coupé ou introuvable (pages {where} → {where_last})")
        else:
            report.append(f"  {label} : entier, page {where[0]}")

    # Traçabilité des nombres.
    unsourced = sorted({x for x in numbers(full) if x not in NOT_CLAIMS and x not in blob})
    if unsourced:
        problems.append(f"{name}: nombres sans source : {unsourced}")
    flat = re.sub(r"\s+", " ", full)
    for phrase in CAUTION[lang]:
        if phrase not in flat:
            problems.append(f"{name}: formule de prudence absente : « {phrase} »")
    for phrase in FORBIDDEN:
        if phrase.lower() in flat.lower():
            problems.append(f"{name}: formulation interdite : « {phrase} »")
    claim, negation = NEGATED[lang]
    for m in re.finditer(re.escape(claim), flat):
        if negation not in flat[max(0, m.start() - 12):m.start()]:
            problems.append(f"{name}: « {claim} » non nié : …{flat[max(0, m.start() - 40):m.end() + 30]}…")
    return problems, report, Counter(numbers(full))


def main() -> int:
    blob = source_blob()
    all_problems = []
    counts = {}
    for lang, name in (("fr", "FR"), ("en", "EN")):
        problems, report, counts[lang] = check(lang, name, blob)
        print("\n".join(report))
        all_problems += problems
    if counts["fr"] != counts["en"]:
        diff = (counts["fr"] - counts["en"]) + (counts["en"] - counts["fr"])
        all_problems.append(f"FR et EN ne portent pas les mêmes nombres : {dict(diff)}")
    else:
        print(f"FR et EN : mêmes nombres ({sum(counts['fr'].values())} occurrences)")
    for p in all_problems:
        print("ÉCART :", p)
    print("OK" if not all_problems else f"{len(all_problems)} écart(s)")
    return 1 if all_problems else 0


if __name__ == "__main__":
    sys.exit(main())
