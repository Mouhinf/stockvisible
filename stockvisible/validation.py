"""Contrat d'entrée : lecture sûre (CSV/Parquet uniquement) et validate(df).

Les règles bloquantes sont les invariants mesurés sur les données réelles
(voir .claude/skills/stockvisible-data-contract/SKILL.md).
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from stockvisible.data import CONTRACT_COLUMNS, SERIES_KEY

ALLOWED_SUFFIXES = frozenset({".csv", ".parquet"})
MAX_BYTES = 20 * 1024 * 1024  # = server.maxUploadSize (Mo) de .streamlit/config.toml
# Plafonds lus AVANT tout décodage (audit M13, P1-1) : un petit fichier compressé ne doit pas pouvoir
# saturer la mémoire d'une instance de 512 Mo. 20 000 lignes = 222 séries de 90 jours (≈ +30 Mo mesurés).
MAX_ROWS = 20_000
MAX_COLUMNS = 64
MAX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
MAX_LIST_CELL_CHARS = 1_000  # une liste JSON de 24 nombres tient largement
HOURS = 24
STOCK_WINDOW = slice(6, 22)
STOCK_WINDOW_HOURS = STOCK_WINDOW.stop - STOCK_WINDOW.start
SUM_TOL = 1e-6
LIST_COLUMNS = ("hours_sale", "hours_stock_status")

# colonne -> (min, max, entier requis) ; None = pas de borne
SCALAR_RULES: dict[str, tuple[float | None, float | None, bool]] = {
    "city_id": (0, None, True),
    "store_id": (0, None, True),
    "management_group_id": (0, None, True),
    "first_category_id": (0, None, True),
    "second_category_id": (0, None, True),
    "third_category_id": (0, None, True),
    "product_id": (0, None, True),
    "sale_amount": (0, None, False),
    "stock_hour6_22_cnt": (0, STOCK_WINDOW_HOURS, True),
    "discount": (0, 1, False),
    "holiday_flag": (0, 1, True),
    "activity_flag": (0, 1, True),
    "precpt": (0, None, False),
    "avg_temperature": (-60, 60, False),
    "avg_humidity": (0, 100, False),
    "avg_wind_level": (0, 17, False),
}

ERROR = "error"
INFO = "info"


class InputRejected(ValueError):
    """Fichier refusé avant toute interprétation de son contenu."""


@dataclass(frozen=True)
class Issue:
    code: str
    severity: str
    message: str
    rows: tuple[int, ...] = ()


@dataclass
class ValidationReport:
    n_rows: int
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == ERROR]

    @property
    def ok(self) -> bool:
        return not self.errors

    def codes(self, severity: str = ERROR) -> set[str]:
        return {i.code for i in self.issues if i.severity == severity}


# ---------------------------------------------------------------- lecture sûre


def read_user_file(path: str | Path, max_bytes: int = MAX_BYTES) -> pd.DataFrame:
    p = Path(path)
    if p.is_symlink() or not p.is_file():
        raise InputRejected(f"{p.name} : pas un fichier ordinaire")
    if p.stat().st_size > max_bytes:
        raise InputRejected(f"{p.name} : taille > {max_bytes} octets")
    return read_user_bytes(p.name, p.read_bytes(), max_bytes)


def read_user_bytes(filename: str, data: bytes, max_bytes: int = MAX_BYTES) -> pd.DataFrame:
    """Lit un envoi utilisateur. Aucun eval/pickle : Parquet via pyarrow, listes CSV via JSON."""
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise InputRejected(f"extension {suffix or '(aucune)'} refusée : CSV ou Parquet uniquement")
    if not data:
        raise InputRejected("fichier vide")
    if len(data) > max_bytes:
        raise InputRejected(f"taille > {max_bytes} octets")
    if suffix == ".parquet":
        return _read_parquet(data)
    return _read_csv(data)


def _check_parquet_schema(schema: pa.Schema) -> None:
    """Colonnes listes : listes de nombres ; autres colonnes : scalaires (jamais struct/map/liste)."""
    for f in schema:
        t = f.type
        if f.name in LIST_COLUMNS:
            is_list = pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t)
            ok = is_list and (pa.types.is_integer(t.value_type) or pa.types.is_floating(t.value_type))
        else:
            ok = not pa.types.is_nested(t)
        if not ok:
            raise InputRejected(f"colonne {f.name} : type {t} non accepté")


def _read_parquet(data: bytes) -> pd.DataFrame:
    if len(data) < 8 or data[:4] != b"PAR1" or data[-4:] != b"PAR1":
        raise InputRejected("contenu non Parquet (signature PAR1 absente)")
    try:
        parquet = pq.ParquetFile(pa.BufferReader(data))
        meta = parquet.metadata  # lu dans le pied de fichier, aucune donnée décodée
        if meta.num_rows > MAX_ROWS:
            raise InputRejected(f"{meta.num_rows} lignes > {MAX_ROWS} au plus")
        if meta.num_columns > MAX_COLUMNS:
            raise InputRejected(f"{meta.num_columns} colonnes > {MAX_COLUMNS} au plus")
        size = sum(meta.row_group(i).total_byte_size for i in range(meta.num_row_groups))
        if size > MAX_UNCOMPRESSED_BYTES:
            raise InputRejected(f"taille décompressée {size} octets > {MAX_UNCOMPRESSED_BYTES}")
        _check_parquet_schema(parquet.schema_arrow)
        return parquet.read().to_pandas()
    except (pa.ArrowException, OSError) as exc:
        raise InputRejected(f"Parquet illisible : {exc}") from exc


def _read_csv(data: bytes) -> pd.DataFrame:
    if b"\x00" in data:
        raise InputRejected("CSV binaire (octet nul)")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InputRejected("CSV non UTF-8") from exc
    lines = text.count("\n")
    if lines > MAX_ROWS + 1:
        raise InputRejected(f"{lines} lignes > {MAX_ROWS} au plus")
    header = text.split("\n", 1)[0]
    if header.count(",") + 1 > MAX_COLUMNS:
        raise InputRejected(f"plus de {MAX_COLUMNS} colonnes")
    try:
        df = pd.read_csv(io.StringIO(text), dtype={"dt": str}, nrows=MAX_ROWS)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise InputRejected(f"CSV illisible : {exc}") from exc
    for col in LIST_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(_parse_json_list)
    return df


def _parse_json_list(cell: object) -> list:
    if not isinstance(cell, str):
        raise InputRejected("cellule liste vide ou non textuelle")
    if len(cell) > MAX_LIST_CELL_CHARS:
        raise InputRejected(f"cellule liste de {len(cell)} caractères > {MAX_LIST_CELL_CHARS}")
    try:
        value = json.loads(cell)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise InputRejected(f"cellule liste non JSON : {cell[:40]!r}") from exc
    if not isinstance(value, list) or not all(
        isinstance(x, (int, float)) and not isinstance(x, bool) for x in value
    ):
        raise InputRejected(f"cellule liste invalide : {cell[:40]!r}")
    return value


# ---------------------------------------------------------------- validation


def _as_sequence(cell: object) -> np.ndarray | None:
    if isinstance(cell, (list, tuple, np.ndarray)):
        try:
            arr = np.asarray(cell, dtype=float)
        except (TypeError, ValueError):
            return None
        return arr if arr.ndim == 1 else None
    return None


def _to_float(raw: pd.Series) -> pd.Series:
    def one(v: object) -> float:
        if isinstance(v, (bool, np.bool_)):
            return float(v)
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                return np.nan
        return np.nan

    try:
        return pd.to_numeric(raw, errors="coerce").astype(float)
    except (TypeError, ValueError):
        return raw.map(one).astype(float)


def validate(df: pd.DataFrame) -> ValidationReport:
    """Contrôle le contrat sans modifier df. Les lignes signalées sont des positions (0..n-1)."""
    frame = df.reset_index(drop=True)
    report = ValidationReport(n_rows=len(frame))

    def flag(code: str, mask, message: str, severity: str = ERROR) -> None:
        rows = np.flatnonzero(np.asarray(mask, dtype=bool))
        if rows.size:
            report.issues.append(Issue(code, severity, message, tuple(int(r) for r in rows)))

    missing = [c for c in CONTRACT_COLUMNS if c not in frame.columns]
    if missing:
        report.issues.append(Issue("MISSING_COLUMN", ERROR, f"colonnes absentes : {missing}"))

    numeric: dict[str, pd.Series] = {}
    for col, (lo, hi, integer) in SCALAR_RULES.items():
        if col not in frame.columns:
            continue
        raw = frame[col]
        values = _to_float(raw)
        flag("NULL_VALUE", raw.isna(), f"{col} : valeur manquante")
        flag("NOT_NUMERIC", raw.notna() & values.isna(), f"{col} : valeur non numérique")
        finite = np.isfinite(values)
        out = values.notna() & ~finite
        if lo is not None:
            out |= finite & (values < lo)
        if hi is not None:
            out |= finite & (values > hi)
        flag("OUT_OF_RANGE", out, f"{col} : hors plage [{lo}, {hi}] ou non fini")
        if integer:
            flag("NON_INTEGER", finite & (values != np.floor(values)), f"{col} : entier attendu")
        numeric[col] = values.where(finite)

    dates = None
    if "dt" in frame.columns:
        dates = pd.to_datetime(frame["dt"], format="%Y-%m-%d", errors="coerce")
        flag("NULL_VALUE", frame["dt"].isna(), "dt : valeur manquante")
        flag("BAD_DATE", frame["dt"].notna() & dates.isna(), "dt : format AAAA-MM-JJ attendu")

    if dates is not None and all(k in frame.columns for k in SERIES_KEY):
        keyed = frame.loc[:, list(SERIES_KEY)].assign(_dt=dates)
        dup = keyed.notna().all(axis=1) & keyed.duplicated(keep=False)
        flag("DUPLICATE_DATE", dup, "même (store_id, product_id, dt) présent plusieurs fois")

    seqs: dict[str, np.ndarray] = {}
    for col in LIST_COLUMNS:
        if col not in frame.columns:
            continue
        parsed = [_as_sequence(c) for c in frame[col]]
        good = np.array(
            [a is not None and a.size == HOURS and bool(np.isfinite(a).all()) for a in parsed],
            dtype=bool,
        )
        flag("BAD_SEQUENCE", ~good, f"{col} : liste de {HOURS} nombres finis attendue")
        stacked = np.full((len(frame), HOURS), np.nan)
        if good.any():
            stacked[good] = np.stack([a for a, g in zip(parsed, good) if g])
        seqs[col] = stacked

    if "hours_sale" in seqs:
        hs = seqs["hours_sale"]
        valid = ~np.isnan(hs).any(axis=1)
        flag("SEQ_OUT_OF_RANGE", valid & (hs < 0).any(axis=1), "hours_sale : valeur négative")
        if "sale_amount" in numeric:
            amount = numeric["sale_amount"].to_numpy()
            both = valid & ~np.isnan(amount)
            gap = np.abs(np.nansum(hs, axis=1) - amount)
            tol = SUM_TOL * np.maximum(1.0, np.abs(np.nan_to_num(amount)))
            flag("SALE_SUM_MISMATCH", both & (gap > tol), "somme(hours_sale) != sale_amount")

    if "hours_stock_status" in seqs:
        st = seqs["hours_stock_status"]
        valid = ~np.isnan(st).any(axis=1)
        binary = valid & np.isin(st, (0.0, 1.0)).all(axis=1)
        flag("SEQ_OUT_OF_RANGE", valid & ~binary, "hours_stock_status : valeurs dans {0, 1} attendues")
        if "stock_hour6_22_cnt" in numeric:
            cnt = numeric["stock_hour6_22_cnt"].to_numpy()
            both = binary & ~np.isnan(cnt)
            window = np.nansum(st[:, STOCK_WINDOW], axis=1)
            flag(
                "STOCK_COUNT_MISMATCH",
                both & (window != cnt),
                "stock_hour6_22_cnt != somme(hours_stock_status[6:22])",
            )
        if "hours_sale" in seqs:
            hs = seqs["hours_sale"]
            during = binary & ((hs > 0) & (st == 1)).any(axis=1)
            flag(
                "SALE_DURING_STOCKOUT",
                during,
                "vente pendant une heure marquée rupture (rupture en cours d'heure, attendu)",
                severity=INFO,
            )

    return report
