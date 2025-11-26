"""Plotly visualization helpers."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .data_processor import MatrixStatistics


def create_cartesian_heatmap(matrix: np.ndarray, stats: MatrixStatistics) -> go.Figure:
    phi_values = np.linspace(stats.phi_min, stats.phi_max, matrix.shape[1])
    q_real_values = np.power(10.0, np.linspace(stats.q_min_exp, stats.q_max_exp, matrix.shape[0]))

    if stats.is_log_transformed:
        display_matrix = np.power(10.0, matrix)
    else:
        display_matrix = matrix

    zmin = float(np.nanmin(display_matrix)) if np.size(display_matrix) else 0.0
    zmax = float(np.nanmax(display_matrix)) if np.size(display_matrix) else 1.0
    if np.isclose(zmin, zmax):
        zmax = zmin + 1e-9

    fig = go.Figure(
        data=go.Heatmap(
            z=display_matrix,
            x=phi_values,
            y=q_real_values,
            colorscale="Inferno",
            colorbar=dict(title="Intensity"),
            zmin=zmin,
            zmax=zmax,
        )
    )
    fig.update_layout(
        title="Cartesian Form",
        xaxis_title="Azimuthal Angle φ (deg)",
        yaxis=dict(
            title="q (Å⁻¹)",
            type="log",
            autorange="reversed",
            tickformat=".2e",
        ),
        template="plotly_white",
        margin=dict(l=80, r=24, t=50, b=60),
    )
    return fig


def create_polar_heatmap(matrix: np.ndarray, stats: MatrixStatistics) -> go.Figure:
    if stats.is_log_transformed:
        display_matrix = np.power(10.0, matrix)
    else:
        display_matrix = matrix

    rows, cols = display_matrix.shape
    mirrored = np.concatenate([display_matrix, np.fliplr(display_matrix)], axis=1)
    extended_cols = mirrored.shape[1]

    zmin = float(np.nanmin(display_matrix)) if np.size(display_matrix) else 0.0
    zmax = float(np.nanmax(display_matrix)) if np.size(display_matrix) else 1.0
    if np.isclose(zmin, zmax):
        zmax = zmin + 1e-9

    canvas_size = max(600, int(max(rows, cols) * 4))
    y_idx, x_idx = np.indices((canvas_size, canvas_size))
    center = (canvas_size - 1) / 2.0
    radius = center * 0.95

    dx = x_idx - center
    dy = center - y_idx
    radial_distance = np.sqrt(dx * dx + dy * dy)
    theta = np.mod(np.arctan2(dy, dx), 2 * np.pi)

    mask = radial_distance <= radius

    row_position = np.clip((radial_distance / radius) * (rows - 1), 0, rows - 1)
    col_position = np.clip((theta / (2 * np.pi)) * (extended_cols - 1), 0, extended_cols - 1)

    row_low = np.floor(row_position).astype(int)
    row_high = np.where(row_low < rows - 1, row_low + 1, row_low)
    row_frac = row_position - row_low

    col_low = np.floor(col_position).astype(int)
    col_high = np.where(col_low < extended_cols - 1, col_low + 1, col_low)
    col_frac = col_position - col_low

    c00 = mirrored[row_low, col_low]
    c01 = mirrored[row_low, col_high]
    c10 = mirrored[row_high, col_low]
    c11 = mirrored[row_high, col_high]

    interpolated = (
        (1.0 - row_frac) * ((1.0 - col_frac) * c00 + col_frac * c01)
        + row_frac * ((1.0 - col_frac) * c10 + col_frac * c11)
    )

    image = np.full((canvas_size, canvas_size), np.nan)
    image[mask] = interpolated[mask]

    q_max_real = float(np.power(10.0, stats.q_max_exp))
    axis_extent = np.linspace(-q_max_real, q_max_real, canvas_size)

    fig = go.Figure(
        data=go.Heatmap(
            z=image,
            x=axis_extent,
            y=axis_extent[::-1],
            colorscale="Inferno",
            colorbar=dict(title="Intensity"),
            zmin=zmin,
            zmax=zmax,
        )
    )
    fig.update_layout(
        title="Polar Form (Projected)",
        xaxis=dict(
            title="q · cos(φ)",
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            title="q · sin(φ)",
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig
