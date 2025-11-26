"""Callbacks responsible for updating figures and summaries."""
from __future__ import annotations

import io
from typing import Dict, Optional

import numpy as np
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots
from dash import Input, Output
from dash import html
import dash

from core.constants import MODEL_PARAMETER_INFO, MODEL_OPTIONS, SUPPORTED_OPTIMIZATION_METHODS
from core.data_processor import MatrixStatistics
from core.visualization import create_cartesian_heatmap, create_polar_heatmap


def _stats_from_dict(stats_dict: Dict[str, float]) -> MatrixStatistics:
    return MatrixStatistics(
        rows=int(stats_dict.get("rows", 0)),
        columns=int(stats_dict.get("columns", 0)),
        min_value=float(stats_dict.get("min_value", 0.0)),
        max_value=float(stats_dict.get("max_value", 0.0)),
        mean=float(stats_dict.get("mean", 0.0)),
        std=float(stats_dict.get("std", 0.0)),
        phi_min=float(stats_dict.get("phi_min", 0.0)),
        phi_max=float(stats_dict.get("phi_max", 180.0)),
        q_min_exp=float(stats_dict.get("q_min_exp", -2.1)),
        q_max_exp=float(stats_dict.get("q_max_exp", -0.9)),
        is_log_transformed=bool(stats_dict.get("is_log_transformed", False)),
    )


def _matrix_from_payload(payload: Dict[str, float]) -> Optional[np.ndarray]:
    data = payload.get("data") if payload else None
    if data is None:
        return None
    return np.array(data, dtype=float)


def _summary_card(label: str, value: str) -> html.Div:
    return html.Div(
        className="data-summary-card",
        children=[html.Div(label, className="card-label"), html.Div(value, className="card-value")],
    )


def register(app):  # noqa: D401
    """Register visualization callbacks."""

    @app.callback(Output("data-summary", "children"), Input("matrix-stats-store", "data"))
    def update_summary(stats_dict):
        if not stats_dict:
            return [html.Div("Upload data to see statistics", className="data-summary-card")]
        stats = _stats_from_dict(stats_dict)
        cards = [
            _summary_card("Matrix Shape", f"{stats.rows} × {stats.columns}"),
            _summary_card("Min", f"{stats.min_value:.4f}"),
            _summary_card("Max", f"{stats.max_value:.4f}"),
            _summary_card("Mean", f"{stats.mean:.4f}"),
            _summary_card("Std Dev", f"{stats.std:.4f}"),
            _summary_card("φ Range", f"{stats.phi_min:.1f}° – {stats.phi_max:.1f}°"),
            _summary_card("log₁₀(q) Range", f"{stats.q_min_exp:.2f} – {stats.q_max_exp:.2f}"),
            _summary_card("Log Transformed", "Yes" if stats.is_log_transformed else "No"),
        ]
        return cards

    @app.callback(
        Output("cartesian-graph", "figure"),
        Input("matrix-store", "data"),
        Input("matrix-stats-store", "data"),
    )
    def update_cartesian(matrix_payload, stats_dict):
        if not matrix_payload or not stats_dict:
            fig = go.Figure()
            fig.update_layout(
                template="plotly_white",
                title="Upload experimental data to generate plots",
            )
            return fig
        matrix = _matrix_from_payload(matrix_payload)
        stats = _stats_from_dict(stats_dict)
        return create_cartesian_heatmap(matrix, stats)

    @app.callback(
        Output("polar-graph", "figure"),
        Input("matrix-store", "data"),
        Input("matrix-stats-store", "data"),
    )
    def update_polar(matrix_payload, stats_dict):
        if not matrix_payload or not stats_dict:
            fig = go.Figure()
            fig.update_layout(
                template="plotly_white",
                title="Upload experimental data to generate plots",
            )
            return fig
        matrix = _matrix_from_payload(matrix_payload)
        stats = _stats_from_dict(stats_dict)
        return create_polar_heatmap(matrix, stats)

    @app.callback(
        Output("scatter-fit-graph", "figure"),
        Input("matrix-store", "data"),
        Input("matrix-stats-store", "data"),
    )
    def update_scatter(matrix_payload, stats_dict):
        if not matrix_payload or not stats_dict:
            fig = go.Figure()
            fig.update_layout(
                template="plotly_white",
                title="Scattering fit will appear after upload and optimization",
            )
            return fig
        matrix = _matrix_from_payload(matrix_payload)
        stats = _stats_from_dict(stats_dict)
        return create_cartesian_heatmap(matrix, stats)

    @app.callback(
        Output("violin-plot-graph", "figure"),
        Input("ga-state-store", "data"),
        Input("crease-results-store", "data"),
    )
    def update_violin(ga_state, uploaded_results_json):
        def build_violin(df: pd.DataFrame) -> go.Figure:
            working_df = df.copy()

            if "Kappa" in working_df.columns and "kappa_log" not in working_df.columns:
                working_df["kappa_log"] = np.log10(pd.to_numeric(working_df["Kappa"], errors="coerce").clip(lower=1e-12))

            ellipsoid_cols = ["MeanR", "StdR", "MeanG", "StdG", "Volume_Fraction"]
            has_ellipsoid = all(col in working_df.columns for col in ellipsoid_cols) and working_df[ellipsoid_cols].notna().any().any()

            if has_ellipsoid:
                subplot_titles = [
                    "Mean Radius (Å)",
                    "Std Radius (Å)",
                    "Mean Aspect Ratio",
                    "Std Aspect Ratio",
                    "log₁₀(Kappa)",
                    "Volume Fraction",
                ]
                fig = make_subplots(
                    rows=2,
                    cols=3,
                    subplot_titles=subplot_titles,
                    horizontal_spacing=0.08,
                    vertical_spacing=0.12,
                )

                params = [
                    ("MeanR", "Mean Radius (Å)"),
                    ("StdR", "Std Radius (Å)"),
                    ("MeanG", "Mean Aspect Ratio"),
                    ("StdG", "Std Aspect Ratio"),
                    ("kappa_log", "log₁₀(Kappa)"),
                    ("Volume_Fraction", "Volume Fraction"),
                ]

                for idx, (col, label) in enumerate(params):
                    series = pd.to_numeric(working_df[col], errors="coerce").dropna()
                    if series.empty:
                        continue
                    row = idx // 3 + 1
                    col_idx = idx % 3 + 1
                    fig.add_trace(
                        go.Violin(
                            y=series,
                            box_visible=True,
                            meanline_visible=True,
                            showlegend=False,
                            name=label,
                        ),
                        row=row,
                        col=col_idx,
                    )

                    min_val = series.min()
                    max_val = series.max()
                    if np.isfinite(min_val) and np.isfinite(max_val):
                        if np.isclose(min_val, max_val):
                            delta = abs(min_val) * 0.1 + 0.1
                            min_val -= delta
                            max_val += delta
                        else:
                            pad = (max_val - min_val) * 0.15
                            min_val -= pad
                            max_val += pad
                        fig.update_yaxes(range=[min_val, max_val], row=row, col=col_idx)

                    fig.update_yaxes(title_text=label, row=row, col=col_idx)

                fig.update_layout(
                    template="plotly_white",
                    title="Ellipsoids Parameter Distributions",
                    showlegend=False,
                    height=900,
                )
                return fig

            subplot_titles = [
                "D_A",
                "omega_deg",
                "alpha_deg",
                "e_mu",
                "e_sigma",
                "kappa_log",
                "dh",
                "lh",
                "nx",
            ]

            fig = make_subplots(
                rows=3,
                cols=3,
                subplot_titles=subplot_titles,
                horizontal_spacing=0.08,
                vertical_spacing=0.12,
            )

            params = [
                ("D_A", "D_A"),
                ("omega_deg", "omega_deg"),
                ("alpha_deg", "alpha_deg"),
                ("e_mu", "e_mu"),
                ("e_sigma", "e_sigma"),
                ("kappa_log", "kappa_log"),
                ("dh", "dh"),
                ("lh", "lh"),
                ("nx", "nx"),
            ]

            for idx, (col, label) in enumerate(params):
                if col not in working_df.columns:
                    continue
                series = pd.to_numeric(working_df[col], errors="coerce").dropna()
                if series.empty:
                    continue

                row = idx // 3 + 1
                col_idx = idx % 3 + 1
                fig.add_trace(
                    go.Violin(
                        y=series,
                        box_visible=True,
                        meanline_visible=True,
                        showlegend=False,
                        name=label,
                    ),
                    row=row,
                    col=col_idx,
                )

                min_val = series.min()
                max_val = series.max()
                if np.isfinite(min_val) and np.isfinite(max_val):
                    if np.isclose(min_val, max_val):
                        delta = abs(min_val) * 0.1 + 0.1
                        min_val -= delta
                        max_val += delta
                    else:
                        pad = (max_val - min_val) * 0.15
                        min_val -= pad
                        max_val += pad
                    fig.update_yaxes(range=[min_val, max_val], row=row, col=col_idx)

                fig.update_yaxes(title_text=label, row=row, col=col_idx)

            fig.update_layout(
                title="Hollow Tubes Parameter Distributions",
                showlegend=False,
                template="plotly_white",
                height=900,
            )
            return fig

        dataframe: Optional[pd.DataFrame] = None

        if uploaded_results_json:
            dataframe = pd.read_json(uploaded_results_json, orient="split")
        elif isinstance(ga_state, dict):
            csv_payload = ga_state.get("csv_results")
            if csv_payload:
                try:
                    dataframe = pd.read_csv(io.StringIO(csv_payload))
                except Exception:
                    dataframe = None

        if dataframe is not None:
            return build_violin(dataframe)

        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            title="Run the GA or upload CREASE results to see parameter distributions",
        )
        return fig

    @app.callback(Output("model-parameters", "children"), Input("model-dropdown", "value"))
    def update_model_parameters(model_value):
        option = next((opt for opt in MODEL_OPTIONS if opt.value == model_value), None)
        params = MODEL_PARAMETER_INFO.get(model_value, [])
        if not option:
            return html.Div("Select a model to see parameter information")
        return html.Div(
            children=[
                html.H4(f"{option.label} parameterization"),
                html.Ul([html.Li(item) for item in params]) if params else html.P("Parameter details coming soon."),
            ]
        )

    method_descriptions = {
        "genetic_algorithm": "Iterative population-based search with crossover, mutation, and elitism.",
        "bayesian_optimization": "Sequential model-based optimization using acquisition functions.",
        "particle_swarm": "Swarm-inspired search balancing exploration and exploitation.",
        "reinforcement_learning": "Policy-driven search adapting to reward feedback.",
    }

    @app.callback(Output("optimization-details", "children"), Input("optimization-method", "value"))
    def update_method_details(method_value):
        description = method_descriptions.get(method_value, "Optimization description unavailable.")
        label = next((opt["label"] for opt in SUPPORTED_OPTIMIZATION_METHODS if opt["value"] == method_value), method_value)
        return html.Div(
            children=[
                html.H4(f"{label}"),
                html.P(description),
            ]
        )

    @app.callback(Output("fit-overview", "children"), Input("ga-state-store", "data"))
    def update_fit_overview(ga_state):
        if not ga_state:
            return html.P("Run the GA to generate fitted scattering profiles.")
        status = ga_state.get("status", "unknown")
        if status == "completed":
            return html.P("Best-fit profile is available for comparison with the experimental data.")
        if status == "failed":
            error = ga_state.get("error") or "Unknown error"
            return html.P(f"Optimization failed: {error}", style={"color": "#dc2626"})
        return html.P("Optimization in progress...")

    @app.callback(Output("ga-result-summary", "children"), Input("ga-state-store", "data"))
    def update_result_summary(ga_state):
        if not ga_state:
            return html.P("Run the genetic algorithm to populate this section.")
        status = ga_state.get("status", "unknown")
        lines = [html.P(f"Status: {status}")]
        if ga_state.get("csv_results"):
            lines.append(html.P("Latest GA results are ready for download."))
        if ga_state.get("error"):
            lines.append(html.P(f"Error: {ga_state['error']}", style={"color": "#dc2626"}))
        return html.Div(lines)

    @app.callback(Output("ga-best-parameters", "children"), Input("ga-state-store", "data"))
    def update_best_parameters(ga_state):
        if not ga_state or ga_state.get("status") != "completed":
            return html.P("Best-fit parameter summary will appear after GA completion.")
        if not ga_state.get("csv_results"):
            return html.P("Download the GA CSV to inspect the best-performing individuals.")
        return html.P("Use the Download GA CSV button to inspect full parameter distributions.")

    @app.callback(Output("summary-table-wrapper", "children"), Input("matrix-stats-store", "data"))
    def update_summary_table(stats_dict):
        if not stats_dict:
            return html.P("Upload data to inspect numerical statistics.")
        stats = _stats_from_dict(stats_dict)
        header = html.Thead(
            html.Tr([html.Th("Statistic"), html.Th("Value")])
        )
        rows = html.Tbody(
            [
                html.Tr([html.Td("Rows"), html.Td(str(stats.rows))]),
                html.Tr([html.Td("Columns"), html.Td(str(stats.columns))]),
                html.Tr([html.Td("Minimum"), html.Td(f"{stats.min_value:.6f}")]),
                html.Tr([html.Td("Maximum"), html.Td(f"{stats.max_value:.6f}")]),
                html.Tr([html.Td("Mean"), html.Td(f"{stats.mean:.6f}")]),
                html.Tr([html.Td("Std Dev"), html.Td(f"{stats.std:.6f}")]),
                html.Tr([html.Td("φ Range"), html.Td(f"{stats.phi_min:.1f}° – {stats.phi_max:.1f}°")]),
                html.Tr([html.Td("log₁₀(q) Range"), html.Td(f"{stats.q_min_exp:.3f} – {stats.q_max_exp:.3f}")]),
                html.Tr([html.Td("Log Transformed"), html.Td("Yes" if stats.is_log_transformed else "No")]),
            ]
        )
        return html.Table([header, rows], className="summary-table")

    @app.callback(
        Output("ga-log-console", "children"),
        Input("ga-state-store", "data"),
        Input("main-tabs", "value"),
    )
    def update_log_console(ga_state, selected_tab):
        # Only update when logs tab is selected to avoid component not found errors
        if selected_tab != "tab-logs":
            raise dash.exceptions.PreventUpdate
        
        if not ga_state:
            return "No GA process running. Start an optimization to see logs here."
        
        logs = ga_state.get("logs", "Waiting for output...")
        return html.Div(logs, style={"white-space": "pre-wrap"})
