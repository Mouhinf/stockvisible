"""F1 DATA — sous-ensemble déterministe et hors-ligne de FreshRetailNet-50K (données réelles)."""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DATASET_REPO = "Dingdong-Inc/FreshRetailNet-50K"
DATASET_REVISION = "08c1fab7f9257bc73679d415d65d644165d351d4"
DATASET_LICENSE = "CC-BY-4.0"
SOURCE_SHA256 = {
    "train": "6706832db892bbae4969c19d87e07975d2543d2ba7d7d4756360654785de5a3d",
    "eval": "1b118840664280c6b88bffc84c80ee1f54c05d911e354b7599e5da10995e960e",
}
SPLITS = ("train", "eval")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

SERIES_KEY = ("store_id", "product_id")
CONTRACT_COLUMNS = (
    "city_id",
    "store_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
    "product_id",
    "dt",
    "sale_amount",
    "hours_sale",
    "stock_hour6_22_cnt",
    "hours_stock_status",
    "discount",
    "holiday_flag",
    "activity_flag",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
)

_DOWNLOAD_TIMEOUT_S = 60
_BATCH_ROWS = 100_000


@dataclass(frozen=True)
class SubsetSpec:
    """Critères figés : tirage uniforme de séries, indépendant des ventes et ruptures."""

    n_series: int = 500
    seed: int = 42


DEFAULT_SPEC = SubsetSpec()


def source_url(split: str) -> str:
    if split not in SPLITS:
        raise ValueError(f"split inconnu : {split!r}")
    return (
        f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/"
        f"{DATASET_REVISION}/data/{split}.parquet"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA-256 inattendu pour {path.name} : {actual} != {expected}")


def download_official(cache_dir: Path = CACHE_DIR) -> dict[str, Path]:
    """Télécharge les fichiers officiels à la révision figée ; réutilise le cache s'il est intègre."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split in SPLITS:
        dest = cache_dir / f"{split}.parquet"
        if not (dest.exists() and sha256_file(dest) == SOURCE_SHA256[split]):
            part = dest.with_suffix(".part")
            with (
                urllib.request.urlopen(source_url(split), timeout=_DOWNLOAD_TIMEOUT_S) as resp,
                open(part, "wb") as fh,
            ):
                shutil.copyfileobj(resp, fh)
            try:
                verify_sha256(part, SOURCE_SHA256[split])
            except ValueError:
                part.unlink()
                raise
            part.replace(dest)
        paths[split] = dest
    return paths


def select_series(keys: pd.DataFrame, n_series: int, seed: int) -> pd.DataFrame:
    """Tirage déterministe de séries, qui ne dépend que des clés (jamais des résultats)."""
    uniq = (
        keys.loc[:, list(SERIES_KEY)]
        .drop_duplicates()
        .sort_values(list(SERIES_KEY))
        .reset_index(drop=True)
    )
    if not 0 < n_series <= len(uniq):
        raise ValueError(f"n_series={n_series} hors de [1, {len(uniq)}]")
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(uniq), size=n_series, replace=False))
    return uniq.iloc[idx].reset_index(drop=True)


def _filter_series(path: Path, selected: pd.DataFrame) -> pa.Table:
    wanted = pd.MultiIndex.from_frame(selected.loc[:, list(SERIES_KEY)])
    pieces = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=_BATCH_ROWS):
        keys = pd.MultiIndex.from_arrays(
            [batch.column(k).to_numpy() for k in SERIES_KEY], names=list(SERIES_KEY)
        )
        mask = keys.isin(wanted)
        if mask.any():
            pieces.append(pa.Table.from_batches([batch]).filter(pa.array(mask)))
    table = pa.concat_tables(pieces)
    return table.sort_by([(k, "ascending") for k in (*SERIES_KEY, "dt")])


def classify_split(train: pd.DataFrame, evaluation: pd.DataFrame) -> dict:
    """Qualifie un split : séries disjointes, découpage temporel des mêmes séries, ou mixte."""
    k_tr = set(map(tuple, train.loc[:, list(SERIES_KEY)].drop_duplicates().to_numpy()))
    k_ev = set(map(tuple, evaluation.loc[:, list(SERIES_KEY)].drop_duplicates().to_numpy()))
    d_tr = pd.to_datetime(train["dt"], format="%Y-%m-%d")
    d_ev = pd.to_datetime(evaluation["dt"], format="%Y-%m-%d")
    shared = k_tr & k_ev
    date_overlap = bool(set(d_tr.unique()) & set(d_ev.unique()))
    if not shared:
        kind = "series_disjoint"
    elif k_tr == k_ev and d_ev.min() > d_tr.max():
        kind = "temporal_same_series"
    else:
        kind = "mixed"
    return {
        "kind": kind,
        "n_series_train": len(k_tr),
        "n_series_eval": len(k_ev),
        "n_series_shared": len(shared),
        "date_overlap": date_overlap,
        "train_dates": [str(d_tr.min().date()), str(d_tr.max().date()), int(d_tr.nunique())],
        "eval_dates": [str(d_ev.min().date()), str(d_ev.max().date()), int(d_ev.nunique())],
    }


def inspect_official_split(cache_dir: Path = CACHE_DIR) -> dict:
    cols = [*SERIES_KEY, "dt"]
    frames = {
        s: pq.read_table(cache_dir / f"{s}.parquet", columns=cols).to_pandas() for s in SPLITS
    }
    return classify_split(frames["train"], frames["eval"])


def _stockout_share_6_22(table: pa.Table) -> float:
    total = pc.sum(table.column("stock_hour6_22_cnt")).as_py()
    return total / (16 * table.num_rows)


def build_subset(
    spec: SubsetSpec = DEFAULT_SPEC,
    cache_dir: Path = CACHE_DIR,
    raw_dir: Path = RAW_DIR,
) -> dict:
    """Construit data/raw/{train,eval}.parquet et manifest.json à partir des fichiers officiels."""
    sources = download_official(cache_dir)
    keys = pq.read_table(sources["train"], columns=list(SERIES_KEY)).to_pandas()
    selected = select_series(keys, spec.n_series, spec.seed)

    raw_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for split in SPLITS:
        table = _filter_series(sources[split], selected)
        out = raw_dir / f"{split}.parquet"
        pq.write_table(table, out)
        outputs[split] = {
            "path": out.name,
            "rows": table.num_rows,
            "sha256": sha256_file(out),
            "stockout_hour_share_6_22": _stockout_share_6_22(table),
        }

    manifest = {
        "data_nature": "REAL — FreshRetailNet-50K, valeurs de vente normalisées (sans unité)",
        "dataset": DATASET_REPO,
        "revision": DATASET_REVISION,
        "license": DATASET_LICENSE,
        "source_sha256": SOURCE_SHA256,
        "subset_spec": asdict(spec),
        "selection_rule": (
            "clés (store_id, product_id) de train triées, tirage uniforme sans remise "
            "numpy.default_rng(seed).choice ; toutes les dates gardées"
        ),
        "outputs": outputs,
        "official_split": inspect_official_split(cache_dir),
    }
    (raw_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def load_raw(split: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Lecture hors-ligne du sous-ensemble local."""
    if split not in SPLITS:
        raise ValueError(f"split inconnu : {split!r}")
    path = raw_dir / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} absent — lancer `python -m stockvisible.data`")
    return pd.read_parquet(path)


def main() -> None:
    manifest = build_subset()
    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
