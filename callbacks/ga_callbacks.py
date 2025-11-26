"""Callbacks that orchestrate GA execution and log streaming."""
from __future__ import annotations

from pathlib import Path

import dash
from dash import Input, Output, State, html

from core.constants import GA_SUPPORTED_MODELS
from core.file_manager import write_upload
from core.ga_manager import GA_MANAGER

GENETIC_ALGORITHM_VALUE = "genetic_algorithm"


def register(app):  # noqa: D401
    """Register callbacks related to GA runs."""

    @app.callback(
        Output("ga-status", "children"),
        Output("ga-state-store", "data"),
        Output("ga-polling-interval", "disabled"),
        Input("run-optimization", "n_clicks"),
        State("matrix-store", "data"),
        State("raw-matrix-store", "data"),
        State("model-dropdown", "value"),
        State("optimization-method", "value"),
        prevent_initial_call=True,
    )
    def start_ga(n_clicks, matrix_payload, raw_matrix_text, model_value, method_value):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate

        if method_value != GENETIC_ALGORITHM_VALUE:
            return (
                html.Span("Only the Genetic Algorithm is available in this build", style={"color": "#dc2626"}),
                dash.no_update,
                True,
            )

        if model_value not in GA_SUPPORTED_MODELS:
            return (
                html.Span("Selected model does not support GA yet", style={"color": "#dc2626"}),
                dash.no_update,
                True,
            )

        if matrix_payload is None or raw_matrix_text is None:
            return (
                html.Span("Upload scattering data before running optimization", style={"color": "#dc2626"}),
                dash.no_update,
                True,
            )

        filename = matrix_payload.get("filename") or "uploaded_scattering_matrix.txt"
        saved_path, safe_name = write_upload(raw_matrix_text.encode("utf-8"), filename, model_value)
        
        # Always reset/kill previous GA processes before starting new one
        GA_MANAGER.reset()
        
        working_dir = Path(__file__).resolve().parents[1] / "genetic_algorithm"
        try:
            GA_MANAGER.start(safe_name, model_value, working_dir)
        except Exception as exc:  # pylint: disable=broad-except
            return (
                html.Span(f"Failed to start GA: {exc}", style={"color": "#dc2626"}),
                dash.no_update,
                True,
            )

        initial_state = {
            "status": "running",
            "logs": "Starting Genetic Algorithm...",
            "csv_results": None,
        }
        status_text = html.Span("Genetic Algorithm running...")
        return status_text, initial_state, False

    @app.callback(
        Output("ga-state-store", "data", allow_duplicate=True),
        Output("ga-status", "children", allow_duplicate=True),
        Output("ga-polling-interval", "disabled", allow_duplicate=True),
        Input("ga-polling-interval", "n_intervals"),
        State("ga-state-store", "data"),
        prevent_initial_call=True,
    )
    def poll_ga(n_intervals, current_state):
        state = GA_MANAGER.get_state()
        logs = state.logs or "Waiting for output..."

        store_payload = {
            "status": state.status,
            "logs": logs,
            "csv_results": state.csv_results,
            "error": state.last_error,
        }

        status_text = html.Span(f"GA status: {state.status}")
        interval_disabled = False

        if state.status in {"completed", "failed"}:
            interval_disabled = True
            if state.status == "failed" and state.last_error:
                status_text = html.Span(f"GA failed: {state.last_error}", style={"color": "#dc2626"})

        return store_payload, status_text, interval_disabled

    @app.callback(
        Output("ga-results-download", "data"),
        Input("download-ga-results-btn", "n_clicks"),
        State("ga-state-store", "data"),
        prevent_initial_call=True,
    )
    def download_ga_csv(n_clicks, ga_state):
        if not n_clicks or not isinstance(ga_state, dict):
            raise dash.exceptions.PreventUpdate

        csv_payload = ga_state.get("csv_results")
        if not csv_payload:
            raise dash.exceptions.PreventUpdate

        return dict(content=csv_payload, filename="crease_ga_results.csv")
