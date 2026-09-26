"""Graphiques Plotly — F2 REVEAL. Deux panneaux à axe x partagé (jamais de double axe y)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from stockvisible.baselines import HOURS, HourlyForecast, hourly_matrix
from ui import theme


def hourly_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    days = pd.to_datetime(frame["dt"], format="%Y-%m-%d").to_numpy()
    hours = np.arange(HOURS) * np.timedelta64(1, "h")
    return pd.DatetimeIndex((days[:, None] + hours).ravel())


def sales_availability_figure(
    train: pd.DataFrame, validation: pd.DataFrame, forecasts: list[HourlyForecast]
) -> go.Figure:
    """Une série : ventes horaires observées + prévisions B0/B1 sur la validation (haut),
    disponibilité déclarée heure par heure (bas)."""
    frame = pd.concat([train, validation]).sort_values("dt", kind="mergesort")
    x = hourly_index(frame)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.08,
        subplot_titles=(
            "Ventes horaires (valeurs normalisées, sans unité)",
            "Disponibilité déclarée (1 = disponible, 0 = rupture)",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=hourly_matrix(frame, "hours_sale").ravel(),
            name="Ventes observées",
            mode="lines",
            line={"color": theme.OBSERVED, "width": 2},
        ),
        row=1,
        col=1,
    )
    for f in forecasts:
        style = theme.FORECAST_STYLE[f.name]
        fig.add_trace(
            go.Scatter(
                x=hourly_index(f.keys),
                y=f.pred.ravel(),
                name=f"{f.name} (prévision validation)",
                mode="lines",
                line={"color": style["color"], "dash": style["dash"], "width": 2},
                connectgaps=False,
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=1 - hourly_matrix(frame, "hours_stock_status").ravel(),
            name="Disponible",
            mode="lines",
            line={"color": theme.NEUTRAL, "width": 1, "shape": "hv"},
            fill="tozeroy",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    start = pd.Timestamp(validation["dt"].min())
    fig.add_vline(x=start, line={"color": theme.NEUTRAL, "dash": "dot", "width": 1})
    fig.add_annotation(
        x=start, y=0.99, xref="x", yref="paper", text="début validation",
        showarrow=False, xanchor="left", font={"size": 11},
    )
    fig.update_yaxes(range=[-0.05, 1.05], tickvals=[0, 1], row=2, col=1)
    fig.update_layout(
        hovermode="x unified",
        height=560,
        margin={"l": 40, "r": 20, "t": 70, "b": 30},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.06, "x": 0},
    )
    return fig
