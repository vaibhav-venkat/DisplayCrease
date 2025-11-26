"""Dynamic tab content rendering."""
from __future__ import annotations

from dash import Input, Output, State, callback, dcc, html


def register(app):  # noqa: D401
    """Register tab layout callback."""

    @app.callback(
        Output("tab-content", "children"),
        Input("main-tabs", "value"),
    )
    def render_tab(selected_tab: str):
        if selected_tab == "tab-data-preview":
            return html.Div(
                style={"margin": "0", "padding": "0"},
                children=[
                    html.Div(id="data-summary", className="data-summary-grid", style={"margin-bottom": "12px"}),
                    dcc.Graph(id="cartesian-graph", style={"height": "420px", "margin-bottom": "12px"}),
                    dcc.Graph(id="polar-graph", style={"height": "520px"}),
                ]
            )
        if selected_tab == "tab-model-workspace":
            return html.Div(
                style={"margin": "0", "padding": "0"},
                children=[
                    html.Div(id="model-parameters", className="results-section", style={"margin-bottom": "12px"}),
                    html.Div(id="optimization-details", className="results-section"),
                ],
            )
        if selected_tab == "tab-scatter":
            return html.Div(
                style={"margin": "0", "padding": "0"},
                children=[
                    html.Div(id="fit-overview", style={"margin-bottom": "8px"}),
                    dcc.Graph(id="scatter-fit-graph", style={"height": "520px", "margin": "0"}),
                ]
            )
        if selected_tab == "tab-distributions":
            return dcc.Graph(id="violin-plot-graph", style={"height": "900px", "margin": "0"})
        if selected_tab == "tab-logs":
            return html.Div(id="ga-log-console", className="log-container", style={"margin": "0"})
        if selected_tab == "tab-results":
            return html.Div(
                style={"margin": "0", "padding": "0"},
                children=[
                    html.Div(
                        style={"display": "flex", "justifyContent": "flex-end", "margin-bottom": "8px"},
                        children=html.Button(
                            "Download GA CSV",
                            id="download-ga-results-btn",
                            className="primary-btn",
                            style={"width": "auto"},
                        ),
                    ),
                    html.Div(id="ga-result-summary", className="results-section", style={"margin-bottom": "12px"}),
                    html.Div(id="ga-best-parameters", className="results-section"),
                ],
            )
        if selected_tab == "tab-summary":
            return html.Div(id="summary-table-wrapper", className="results-section", style={"margin": "0"})
        if selected_tab == "tab-reconstruction":
            return html.Div(
                className="results-section",
                style={"margin": "0"},
                children="3D real-space reconstruction coming soon. Upload GA outputs to preview.",
            )
        return html.Div("Select a tab to continue")
