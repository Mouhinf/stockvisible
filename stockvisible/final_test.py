"""Test final — UNE seule exécution (M9). Protocole écrit AVANT l'ouverture du test.

Décisions utilisateur, prises avant l'ouverture :
- moteur exécuté = modèle gelé en M8, entraîné sur le TRAIN seul (aucun réentraînement) ;
- ML, B1 et B0 évalués dans le même passage, avec le protocole de la validation : prévision
  glissante à J+1 sur les 15 jours de test (l'historique observé = train + validation + jours de
  test < J), heures disponibles, mêmes cellules, MAE et biais. Aucune décision n'en dépend.

Garde-fous : le gel doit être intact (`check_freeze`), le fichier de résultat est créé en mode
exclusif (une seconde exécution échoue), et le test scellé n'est ouvert qu'une fois, ici.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from stockvisible.baselines import b0_forecast, b1_forecast
from stockvisible.data import PROJECT_ROOT, RAW_DIR, load_raw, sha256_file
from stockvisible.evaluation import PRIMARY_SCOPE, compare_baselines, rolling_origin
from stockvisible.model import FittedModel, fit_model, ml_forecast
from stockvisible.selection import FREEZE_PATH, check_freeze, load_freeze
from stockvisible.splits import UNSEAL_PHRASE, chronological_split

RESULT_PATH = PROJECT_ROOT / "logs" / "final_test_result.json"
PROTOCOL = {
    "entraînement": "modèle gelé M8, train seul (2024-03-28 → 2024-05-26), aucun réentraînement",
    "prévision": "glissante J+1 ; historique = train + validation + jours de test < J",
    "périmètre_principal": PRIMARY_SCOPE,
    "métriques": "MAE et biais (moyenne prévision − réel), mêmes cellules pour ML, B1, B0",
    "modèles": ["ML (moteur gelé)", "B1 (contexte)", "B0 (contexte)"],
}


class FinalTestAlreadyRun(RuntimeError):
    """Le test final a déjà été exécuté : on ne le relance jamais."""


def evaluate_sealed(
    history: pd.DataFrame, targets: pd.DataFrame, model: FittedModel, label: str
) -> pd.DataFrame:
    """Même protocole que la validation M8, sur n'importe quelle période cible postérieure."""
    forecasts = [
        rolling_origin(b0_forecast, "B0 (glissant)")(history, targets),
        rolling_origin(b1_forecast, "B1 (glissant)")(history, targets),
        replace(ml_forecast(model, history, targets), name="ML (glissant)"),
    ]
    table = compare_baselines(targets, forecasts, label=label)
    return table.assign(dt_min=targets["dt"].min(), dt_max=targets["dt"].max())


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def run_final_test(result_path: Path = RESULT_PATH, raw_dir: Path = RAW_DIR) -> dict:
    if result_path.exists():
        raise FinalTestAlreadyRun(f"{result_path} existe déjà : le test final a eu lieu")
    freeze = load_freeze()
    check_freeze(freeze)  # avant toute lecture : la configuration doit être celle de M8
    started = datetime.now(UTC).isoformat(timespec="seconds")

    splits = chronological_split(load_raw("train", raw_dir=raw_dir))
    model = fit_model(splits.train)
    if (model.train_start, model.train_end) != tuple(freeze["périodes"]["train"]):
        raise RuntimeError("le modèle réentraîné ne couvre pas la période gelée")
    if model.n_train_rows != freeze["modèle"]["n_lignes_train"]:
        raise RuntimeError("le modèle réentraîné ne voit pas les mêmes lignes que le modèle gelé")

    test = splits.test.open(UNSEAL_PHRASE)  # SEULE ouverture du test réservé
    if str(test["dt"].min()) != freeze["périodes"]["début_test_réservé"]:
        raise RuntimeError("le test ne commence pas à la date réservée")
    history = pd.concat([splits.train, splits.validation], ignore_index=True)
    table = evaluate_sealed(history, test, model, label="test")

    primary = table[table["périmètre"] == PRIMARY_SCOPE].set_index("modèle")
    engine_name = f"{freeze['moteur']} (glissant)"

    def row(name: str) -> dict:
        return {k: float(primary.at[name, k]) for k in ("mae", "bias", "couverture")} | {
            "n": int(primary.at[name, "n"])
        }

    record = {
        "horodatage_début_utc": started,
        "horodatage_fin_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "ouvertures_du_test": 1,
        "moteur_gelé": freeze["moteur"],
        "résultat_moteur_test": row(engine_name),
        "contexte_test": {name: row(name) for name in primary.index if name != engine_name},
        "rappel_validation_M8": freeze["métriques_validation_périmètre_principal"],
        "protocole": PROTOCOL,
        "période_test": [str(test["dt"].min()), str(test["dt"].max())],
        "lignes_test": len(test),
        "séries": int(test[["store_id", "product_id"]].drop_duplicates().shape[0]),
        "table_test": json.loads(table.to_json(orient="records", force_ascii=False)),
        "provenance": {
            "git_head": _git("rev-parse", "HEAD"),
            "fichiers_non_commités_au_lancement": _git("status", "--porcelain").splitlines(),
            "engine_freeze_sha256": sha256_file(FREEZE_PATH),
            "final_test_py_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "train_parquet_sha256": sha256_file(raw_dir / "train.parquet"),
        },
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "x", encoding="utf-8") as fh:  # "x" : jamais d'écrasement
        fh.write(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    return record


def main() -> None:
    record = run_final_test()
    print(json.dumps({k: record[k] for k in ("horodatage_fin_utc", "moteur_gelé",
                                             "résultat_moteur_test", "contexte_test")},
                     indent=2, ensure_ascii=False))
    print(f"écrit : {RESULT_PATH}")


if __name__ == "__main__":
    main()
