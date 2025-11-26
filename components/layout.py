"""Dash layout definitions for the integrated CREASE-2D application."""
from __future__ import annotations

from dash import dcc, html

from core.constants import MODEL_OPTIONS, SUPPORTED_OPTIMIZATION_METHODS


def _build_header() -> html.Div:
    return html.Div(
        className="app-header",
        children=[
            html.H1("CREASE-2D Integrated Workbench", className="app-title"),
            html.P(
                "Unified interface for scattering data analysis, model selection, and GA optimization",
                className="app-subtitle",
            ),
        ],
    )


def _build_sidebar() -> html.Div:
    return html.Div(
        className="sidebar",
        children=[
            html.H3("Data Upload", className="sidebar-section-title"),
            html.Label("Experimental Scattering Matrix", className="sidebar-label"),
            dcc.Upload(
                id="upload-matrix",
                children=html.Div(["Drag and Drop or ", html.A("Select File")]),
                multiple=False,
                className="upload-area",
            ),
            html.Div(id="upload-feedback", className="upload-feedback"),
            html.Label("CREASE GA Results (CSV)", className="sidebar-label"),
            dcc.Upload(
                id="upload-crease-results",
                children=html.Div(["Drag and Drop or ", html.A("Select File")]),
                multiple=False,
                className="upload-area",
            ),
            html.Div(id="crease-upload-feedback", className="upload-feedback"),
            html.Hr(),
            html.H4("Data Configuration", className="sidebar-section-title"),
            html.Div(
                className="config-grid",
                children=[
                    _number_input("phi-min-input", "φ Min", 0, step=1),
                    _number_input("phi-max-input", "φ Max", 180, step=1),
                    _number_input("q-min-input", "log₁₀(q) Min", -2.1, step=0.1),
                    _number_input("q-max-input", "log₁₀(q) Max", -0.9, step=0.1),
                ],
            ),
            html.Label(
                className="checkbox-label",
                children=[
                    dcc.Checklist(
                        id="log-transform-toggle",
                        options=[{"label": "Data is log-transformed", "value": "log"}],
                        value=[],
                    ),
                ],
            ),
            html.Button("Recompute Plots", id="recompute-plots", className="primary-btn"),
            html.Hr(),
            html.H3("Model & Optimization", className="sidebar-section-title"),
            dcc.Dropdown(
                id="model-dropdown",
                options=[{"label": opt.label, "value": opt.value} for opt in MODEL_OPTIONS],
                value="hollow_tubes",
                clearable=False,
                className="sidebar-dropdown",
            ),
            dcc.RadioItems(
                id="optimization-method",
                options=SUPPORTED_OPTIMIZATION_METHODS,
                value="genetic_algorithm",
                className="sidebar-radio-group",
            ),
            html.Button("Run Optimization", id="run-optimization", className="primary-btn"),
            html.Div(id="ga-status", className="status-text"),
        ],
    )


def _number_input(component_id: str, label: str, value: float, step: float) -> html.Div:
    return html.Div(
        className="number-input-group",
        children=[
            html.Label(label, htmlFor=component_id, style={"fontSize": "12px", "fontWeight": "500"}),
            dcc.Input(
                id=component_id,
                type="number",
                value=value,
                step=step,
                debounce=True,
                className="number-input",
                style={"width": "100%", "minWidth": "0"},
            ),
        ],
    )


def _build_tabs() -> html.Div:
    return html.Div(
        className="main-content",
        children=[
            dcc.Tabs(
                id="main-tabs",
                value="tab-data-preview",
                children=[
                    dcc.Tab(label="Data Preview", value="tab-data-preview"),
                    dcc.Tab(label="Model Workspace", value="tab-model-workspace"),
                    dcc.Tab(label="Scattering Fit", value="tab-scatter"),
                    dcc.Tab(label="Parameter Distributions", value="tab-distributions"),
                    dcc.Tab(label="Processing Logs", value="tab-logs"),
                    dcc.Tab(label="Results", value="tab-results"),
                    dcc.Tab(label="Summary", value="tab-summary"),
                    dcc.Tab(label="Real-Space Reconstruction", value="tab-reconstruction"),
                ],
            ),
            html.Div(id="tab-content", className="tab-content"),
        ],
    )


def build_layout() -> html.Div:
    return html.Div(
        id="app-container",
        children=[
            dcc.Store(id="raw-matrix-store"),
            dcc.Store(id="matrix-store"),
            dcc.Store(id="matrix-stats-store"),
            dcc.Store(id="crease-results-store"),
            dcc.Store(id="ga-state-store"),
            dcc.Interval(id="ga-polling-interval", interval=2000, disabled=True),
            dcc.Download(id="ga-results-download"),
            _build_header(),
            html.Div(
                className="content-wrapper",
                children=[_build_sidebar(), _build_tabs()],
            ),
        ],
    )
