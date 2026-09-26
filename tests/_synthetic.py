"""Données SYNTHÉTIQUES, réservées aux tests. Jamais présentées comme réelles."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockvisible.data import CONTRACT_COLUMNS

SYNTHETIC_LABEL = "SYNTHETIC-TEST-ONLY"


def make_valid_frame(n_series: int = 3, n_days: int = 4, seed: int = 0) -> pd.DataFrame:
    """Frame cohérent avec tous les invariants du contrat, dont les cas réels limites."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_series):
        for d in range(n_days):
            status = (rng.random(24) < 0.25).astype(int)
            sale = np.round(rng.random(24) * 2, 1) * (1 - status)
            rows.append(
                {
                    "city_id": 1,
                    "store_id": 10 + s,
                    "management_group_id": 0,
                    "first_category_id": 5,
                    "second_category_id": 50,
                    "third_category_id": 500,
                    "product_id": 100 + s,
                    "dt": str((pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)).date()),
                    "sale_amount": float(sale.sum()),
                    "hours_sale": sale.tolist(),
                    "stock_hour6_22_cnt": int(status[6:22].sum()),
                    "hours_stock_status": status.tolist(),
                    "discount": 1.0,
                    "holiday_flag": 0,
                    "activity_flag": int(d % 2),
                    "precpt": 0.5,
                    "avg_temperature": 20.0,
                    "avg_humidity": 60.0,
                    "avg_wind_level": 1.5,
                }
            )
    df = pd.DataFrame(rows, columns=list(CONTRACT_COLUMNS))

    # Cas réel limite 1 : vente nulle SANS rupture (≠ rupture).
    df.at[0, "hours_sale"] = [0.0] * 24
    df.at[0, "sale_amount"] = 0.0
    df.at[0, "hours_stock_status"] = [0] * 24
    df.at[0, "stock_hour6_22_cnt"] = 0
    # Cas réel limite 2 : vente pendant une heure marquée rupture (rupture en cours d'heure).
    sale = [0.0] * 24
    sale[10] = 1.5
    status = [0] * 24
    status[10] = 1
    df.at[1, "hours_sale"] = sale
    df.at[1, "sale_amount"] = 1.5
    df.at[1, "hours_stock_status"] = status
    df.at[1, "stock_hour6_22_cnt"] = 1
    return df
