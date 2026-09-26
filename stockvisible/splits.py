"""Split chronologique par série : 60 jours train / 15 validation / 15 test. Jamais aléatoire."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockvisible.data import SERIES_KEY

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


def chronological_split(df: pd.DataFrame, spec: SplitSpec = SPLIT_SPEC) -> Splits:
    """Chaque série doit couvrir exactement spec.total_days jours consécutifs."""
    keys = list(SERIES_KEY)
    dates = pd.to_datetime(df["dt"], format="%Y-%m-%d")
    ordered = df.assign(_date=dates).sort_values([*keys, "_date"], kind="mergesort")

    per_series = ordered.groupby(keys, sort=False)["_date"]
    n_days = per_series.transform("size")
    step = per_series.diff().dt.days
    bad_len = ordered.loc[n_days != spec.total_days, keys].drop_duplicates()
    bad_gap = ordered.loc[step.notna() & (step != 1), keys].drop_duplicates()
    if len(bad_len) or len(bad_gap):
        raise ValueError(
            f"séries hors contrat ({spec.total_days} jours consécutifs sans doublon) : "
            f"longueur {bad_len.to_numpy().tolist()[:5]}, trou/doublon {bad_gap.to_numpy().tolist()[:5]}"
        )

    rank = per_series.cumcount().to_numpy()
    part = np.select(
        [rank < spec.train_days, rank < spec.train_days + spec.validation_days],
        ["train", "validation"],
        default="test",
    )
    ordered = ordered.drop(columns="_date")

    def take(name: str) -> pd.DataFrame:
        return ordered.loc[part == name].reset_index(drop=True)

    return Splits(train=take("train"), validation=take("validation"), test=SealedFrame(take("test")))


def date_ranges(splits: Splits) -> dict[str, tuple[str, str]]:
    """Bornes de dates par partition (le test n'expose que ses bornes, pas ses valeurs)."""
    out = {name: (frame["dt"].min(), frame["dt"].max()) for name, frame in
           (("train", splits.train), ("validation", splits.validation))}
    test = splits.test._df
    out["test"] = (test["dt"].min(), test["dt"].max())
    return out
