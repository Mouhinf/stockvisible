"""F3 VERIFY — choix du moteur sur la VALIDATION et gel de sa configuration (M8).

Toute la chaîne passe par `load_dev()` : la période test n'est jamais construite. La décision et
tout ce qui la rend reproductible sont écrits dans `engine_freeze.json`. Après ce gel,
`model.py`, `features.py` et `baselines.py` (le code que le moteur exécute) ne doivent plus
changer : `check_freeze()` compare leurs SHA-256 à ceux du gel.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import sklearn

from stockvisible.data import PROJECT_ROOT, RAW_DIR, sha256_file
from stockvisible.evaluation import (
    PRIMARY_SCOPE,
    SELECTION_RULE,
    rolling_validation_table,
    select_engine,
)
from stockvisible.features import FEATURE_SPEC, FEATURES
from stockvisible.model import MODEL_SPEC, FittedModel, fit_model
from stockvisible.splits import SPLIT_SPEC, DevSplits, load_dev, reserved_period_start

FREEZE_PATH = PROJECT_ROOT / "engine_freeze.json"
FROZEN_FILES = ("stockvisible/model.py", "stockvisible/features.py", "stockvisible/baselines.py")


class FreezeViolation(AssertionError):
    """Le code ou la configuration du moteur a changé après le gel."""


def file_hashes(root: Path = PROJECT_ROOT) -> dict[str, str]:
    return {rel: hashlib.sha256((root / rel).read_bytes()).hexdigest() for rel in FROZEN_FILES}


def run_selection(dev: DevSplits | None = None) -> tuple[dict, pd.DataFrame, FittedModel, str]:
    """Entraîne sur train, compare B0/B1/ML en glissant sur la validation, applique la règle."""
    dev = dev if dev is not None else load_dev()
    start = reserved_period_start(dev)
    model = fit_model(dev.train)
    table = rolling_validation_table(dev, model)
    return select_engine(table, test_start=start), table, model, start


def freeze_record(decision: dict, table: pd.DataFrame, model: FittedModel, start: str) -> dict:
    primary = table[table["périmètre"] == PRIMARY_SCOPE]
    metrics = {
        row["modèle"]: {"mae": float(row["mae"]), "bias": float(row["bias"]), "n": int(row["n"]),
                        "couverture": float(row["couverture"])}
        for _, row in primary.iterrows()
    }
    return {
        "moteur": decision["moteur"],
        "règle": SELECTION_RULE,
        "décision": {"mae_ml": float(decision["mae_ml"]), "mae_b1": float(decision["mae_b1"]),
                     "n": int(decision["n"])},
        "métriques_validation_périmètre_principal": metrics,
        "table_validation": json.loads(table.to_json(orient="records", force_ascii=False)),
        "périodes": {
            "train": [model.train_start, model.train_end],
            "validation": [str(table["dt_min"].min()), str(table["dt_max"].max())],
            "début_test_réservé": start,
        },
        "regards_sur_la_validation": (
            "2 regards déclarés sur le ML : M7 (ML dégénéré) puis M7b (correctif de départage). "
            "M8 recalcule la même configuration, sans aucun changement."
        ),
        "modèle": {"spec": asdict(MODEL_SPEC), "n_lignes_train": model.n_train_rows,
                   "sklearn": sklearn.__version__, "python": platform.python_version()},
        "variables": {"spec": asdict(FEATURE_SPEC), "liste": list(FEATURES)},
        "split": asdict(SPLIT_SPEC),
        "données": {"train_parquet_sha256": sha256_file(RAW_DIR / "train.parquet")},
        "fichiers_gelés_sha256": file_hashes(),
        "entraînement_final": "à décider AVANT d'ouvrir le test (train seul, ou train + validation)",
    }


def write_freeze(record: dict, path: Path = FREEZE_PATH) -> None:
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False, sort_keys=False) + "\n", encoding="utf-8")


def load_freeze(path: Path = FREEZE_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check_freeze(record: dict, root: Path = PROJECT_ROOT) -> None:
    """Lève FreezeViolation si un fichier gelé ou une spécification a changé depuis le gel."""
    problems = []
    current = file_hashes(root)
    for rel, digest in record["fichiers_gelés_sha256"].items():
        if current.get(rel) != digest:
            problems.append(f"{rel} modifié après le gel")
    if record["modèle"]["spec"] != asdict(MODEL_SPEC):
        problems.append("ModelSpec modifié après le gel")
    if record["variables"]["spec"] != json.loads(json.dumps(asdict(FEATURE_SPEC))):
        problems.append("FeatureSpec modifié après le gel")
    if record["variables"]["liste"] != list(FEATURES):
        problems.append("liste des variables modifiée après le gel")
    if problems:
        raise FreezeViolation(" ; ".join(problems))


def main() -> None:
    decision, table, model, start = run_selection()
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(table[table["périmètre"] == PRIMARY_SCOPE].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"))
    print(decision)
    write_freeze(freeze_record(decision, table, model, start))
    print(f"gel écrit : {FREEZE_PATH}")


if __name__ == "__main__":
    main()
