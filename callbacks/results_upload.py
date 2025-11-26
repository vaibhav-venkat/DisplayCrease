"""Callbacks for ingesting CREASE GA result tables."""
from __future__ import annotations

import base64
import io
from typing import Optional

import dash
import pandas as pd
from dash import Input, Output, State, html


def _decode_to_dataframe(contents: str, filename: str) -> pd.DataFrame:
    _, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    buffer = io.StringIO(decoded.decode("utf-8"))
    if filename.lower().endswith((".txt", ".dat")):
        return pd.read_csv(buffer, sep="\s+")
    return pd.read_csv(buffer)


def register(app):
    @app.callback(
        Output("crease-results-store", "data"),
        Output("crease-upload-feedback", "children"),
        Input("upload-crease-results", "contents"),
        State("upload-crease-results", "filename"),
        prevent_initial_call=True,
    )
    def on_results_upload(contents: Optional[str], filename: Optional[str]):
        if contents is None or filename is None:
            raise dash.exceptions.PreventUpdate
        try:
            df = _decode_to_dataframe(contents, filename)
            payload = df.to_json(orient="split", date_format="iso")
            feedback = html.Span(["Loaded results: ", html.B(filename)])
            return payload, feedback
        except Exception as exc:  # pylint: disable=broad-except
            feedback = html.Span(f"Failed to parse {filename}: {exc}", style={"color": "#dc2626"})
            return dash.no_update, feedback
