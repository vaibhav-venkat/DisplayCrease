"""Integrated Dash application that merges CREASE-2D frontends."""
from __future__ import annotations

import dash
from dash import Dash

from callbacks import register_callbacks
from components.layout import build_layout


def create_app() -> Dash:
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    app.title = "CREASE-2D Integrated"
    app.layout = build_layout()
    register_callbacks(app)
    return app


app = create_app()
server = app.server


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
