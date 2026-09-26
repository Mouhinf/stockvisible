"""Split chronologique par série : 60 jours train / 15 validation / 15 test. Jamais aléatoire."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from stockvisible.data import RAW_DIR, SERIES_KEY

UNSEAL_PHRASE = "TEST FINAL — moteur figé"


@dataclass(frozen=True)
class SplitSpec:
    """Fixé avant tout calcul de performance (M2)."""

    train_days: int = 60
    validation_days: int = 15
    test_days: int = 15

    @property
    def total_days(self) -> int:
        return self.train_days + self.validation_days + self.test_days


SPLIT_SPEC = SplitSpec()


class SealedFrame:
    """Garde-fou contre l'usage accidentel du test : ouverture explicite uniquement."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    @property
    def n_rows(self) -> int:
        return len(self._df)

    def open(self, confirm: str) -> pd.DataFrame:
        if confirm != UNSEAL_PHRASE:
            raise PermissionError("test final scellé : réservé au moteur figé, une seule fois")
        return self._df.copy()

    def __repr__(self) -> str:
        return f"<SealedFrame test final scellé, {self.n_rows} lignes>"


@dataclass(frozen=True)
class Splits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: SealedFrame


@dataclass(frozen=True)
class DevSplits:
    """Train + validation uniquement : aucun attribut ne donne accès à la période test."""

    train: pd.DataFrame
    validation: pd.DataFrame


def _ordered_with_contract(df: pd.DataFrame, n_days: int) -> tuple[pd.DataFrame, np.ndarray]:
    keys = list(SERIES_KEY)
    dates = pd.to_datetime(df["dt"], format="%Y-%m-%d")
    ordered = df.assign(_date=dates).sort_values([*keys, "_date"], kind="mergesort")
    per_series = ordered.groupby(keys, sort=False)["_date"]
    size = per_series.transform("size")
    step = per_series.diff().dt.days
    bad_len = ordered.loc[size != n_days, keys].drop_duplicates()
    bad_gap = ordered.loc[step.notna() & (step != 1), keys].drop_duplicates()
    if len(bad_len) or len(bad_gap):
        raise ValueError(
            f"séries hors contrat ({n_days} jours consécutifs sans doublon) : "
            f"longueur {bad_len.to_numpy().tolist()[:5]}, "
            f"trou/doublon {bad_gap.to_numpy().tolist()[:5]}"
        )
    return ordered, per_series.cumcount().to_numpy()


def chronological_split(df: pd.DataFrame, spec: SplitSpec = SPLIT_SPEC) -> Splits:
    """Chaque série doit couvrir exactement spec.total_days jours consécutifs."""
    ordered, rank = _ordered_with_contract(df, spec.total_days)
    part = np.select(
        [rank < spec.train_days, rank < spec.train_days + spec.validation_days],
        ["train", "validation"],
        default="test",
    )
    ordered = ordered.drop(columns="_date")

    def take(name: str) -> pd.DataFrame:
        return ordered.loc[part == name].reset_index(drop=True)

    return Splits(train=take("train"), validation=take("validation"), test=SealedFrame(take("test")))


def dev_split(df_dev: pd.DataFrame, spec: SplitSpec = SPLIT_SPEC) -> DevSplits:
    """df_dev ne contient que les jours dev : train_days + validation_days par série."""
    ordered, rank = _ordered_with_contract(df_dev, spec.train_days + spec.validation_days)
    ordered = ordered.drop(columns="_date")
    is_train = rank < spec.train_days
    return DevSplits(
        train=ordered.loc[is_train].reset_index(drop=True),
        validation=ordered.loc[~is_train].reset_index(drop=True),
    )


def load_dev(raw_dir: Path = RAW_DIR, spec: SplitSpec = SPLIT_SPEC) -> DevSplits:
    """Charge train + validation sans jamais matérialiser la période test.

    Seules les colonnes clés et dt sont lues sur les 90 jours (pour vérifier le contrat et
    placer la coupure). Les lignes de la période test sont ensuite écartées au niveau Arrow,
    par lot, avant toute conversion : aucune valeur de test n'est jamais renvoyée.
    """
    path = raw_dir / "train.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} absent — lancer `python -m stockvisible.data`")
    keys = list(SERIES_KEY)
    calendar = pq.read_table(path, columns=[*keys, "dt"]).to_pandas()
    ordered, rank = _ordered_with_contract(calendar, spec.total_days)
    last_dev = ordered.loc[rank == spec.train_days + spec.validation_days - 1, [*keys, "_date"]]

    pieces = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=100_000):
        k = batch.select([*keys, "dt"]).to_pandas()
        limit = k.merge(last_dev, on=keys, how="left", validate="many_to_one")["_date"]
        keep = (pd.to_datetime(k["dt"], format="%Y-%m-%d") <= limit).to_numpy()
        pieces.append(pa.Table.from_batches([batch]).filter(pa.array(keep)).to_pandas())
    return dev_split(pd.concat(pieces, ignore_index=True), spec)


def date_ranges(splits: Splits) -> dict[str, tuple[str, str]]:
    """Bornes de dates par partition (le test n'expose que ses bornes, pas ses valeurs)."""
    out = {name: (frame["dt"].min(), frame["dt"].max()) for name, frame in
           (("train", splits.train), ("validation", splits.validation))}
    test = splits.test._df
    out["test"] = (test["dt"].min(), test["dt"].max())
    return out
