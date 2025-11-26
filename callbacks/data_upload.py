"""Callbacks related to file ingestion and data processing."""
from __future__ import annotations

import base64
from typing import Optional

import dash
from dash import Input, Output, State, exceptions, html

from core.data_processor import MatrixProcessingError, parse_matrix_text


def _decode_upload(contents: str) -> str:
    content_type, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    return decoded.decode("utf-8")


def register(app):  # noqa: D401
    """Register upload related callbacks with the Dash app."""

    @app.callback(
        Output("raw-matrix-store", "data"),
        Output("matrix-store", "data"),
        Output("matrix-stats-store", "data"),
        Output("upload-feedback", "children"),
        Input("upload-matrix", "contents"),
        State("upload-matrix", "filename"),
        State("phi-min-input", "value"),
        State("phi-max-input", "value"),
        State("q-min-input", "value"),
        State("q-max-input", "value"),
        State("log-transform-toggle", "value"),
        
        prevent_initial_call=True,
    )
    def on_upload(contents: Optional[str], filename: Optional[str], phi_min, phi_max, q_min, q_max, toggle):
        if contents is None or filename is None:
            raise exceptions.PreventUpdate

        try:
            raw_text = _decode_upload(contents)
            user_config = {
                "phi_min": phi_min,
                "phi_max": phi_max,
                "q_min_exp": q_min,
                "q_max_exp": q_max,
                "is_log_transformed": bool(toggle and "log" in toggle),
            }
            processed = parse_matrix_text(raw_text, user_config=user_config)
            stats_dict = processed.statistics.to_dict()
            matrix_payload = {
                "data": processed.matrix.tolist(),
                "phi_min": stats_dict["phi_min"],
                "phi_max": stats_dict["phi_max"],
                "q_min_exp": stats_dict["q_min_exp"],
                "q_max_exp": stats_dict["q_max_exp"],
                "is_log_transformed": stats_dict["is_log_transformed"],
                "filename": filename,
            }
            feedback = html.Span(["Loaded: ", html.B(filename)])
            return raw_text, matrix_payload, stats_dict, feedback
        except MatrixProcessingError as exc:
            feedback = html.Span(str(exc), style={"color": "#dc2626"})
            return dash.no_update, dash.no_update, dash.no_update, feedback

    @app.callback(
        Output("matrix-store", "data", allow_duplicate=True),
        Output("matrix-stats-store", "data", allow_duplicate=True),
        Output("upload-feedback", "children", allow_duplicate=True),
        Input("recompute-plots", "n_clicks"),
        State("raw-matrix-store", "data"),
        State("matrix-store", "data"),
        State("phi-min-input", "value"),
        State("phi-max-input", "value"),
        State("q-min-input", "value"),
        State("q-max-input", "value"),
        State("log-transform-toggle", "value"),
        prevent_initial_call=True,
    )
    def recompute(n_clicks, raw_text, matrix_payload, phi_min, phi_max, q_min, q_max, toggle):
        if not n_clicks:
            raise exceptions.PreventUpdate
        if raw_text is None:
            return dash.no_update, dash.no_update, html.Span("Upload a file to process", style={"color": "#2563eb"})

        try:
            user_config = {
                "phi_min": phi_min,
                "phi_max": phi_max,
                "q_min_exp": q_min,
                "q_max_exp": q_max,
                "is_log_transformed": bool(toggle and "log" in toggle),
            }
            processed = parse_matrix_text(raw_text, user_config=user_config)
            stats_dict = processed.statistics.to_dict()
            filename = None
            if matrix_payload and matrix_payload.get("filename"):
                filename = matrix_payload["filename"]
            matrix_payload = {
                "data": processed.matrix.tolist(),
                "phi_min": stats_dict["phi_min"],
                "phi_max": stats_dict["phi_max"],
                "q_min_exp": stats_dict["q_min_exp"],
                "q_max_exp": stats_dict["q_max_exp"],
                "is_log_transformed": stats_dict["is_log_transformed"],
                "filename": filename,
            }
            feedback = html.Span("Plots updated with current configuration", style={"color": "#16a34a"})
            return matrix_payload, stats_dict, feedback
        except MatrixProcessingError as exc:
            feedback = html.Span(str(exc), style={"color": "#dc2626"})
            return dash.no_update, dash.no_update, feedback
