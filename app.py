# FILE: app.py

import dash
from dash import dcc, html

from callbacks.load_data import register_callbacks as register_load_data_callbacks
from callbacks.plotting import register_callbacks as register_plotting_callbacks

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

DEFAULT_UPLOAD_CHILDREN = html.Div(["Drag and Drop or ", html.A("Select File")])

app.layout = html.Div(
    id="main-container",
    style={"fontFamily": "sans-serif"},
    children=[
        dcc.Store(id="experimental-data-store"),
        dcc.Store(id="crease-results-store"),
        html.H1(
            "DisplayCREASE",
            style={"textAlign": "center", "margin": "20px"},
        ),
        html.Div(
            id="content-container",
            style={"display": "flex", "height": "calc(100vh - 80px)"},
            children=[
                html.Div(
                    id="sidebar",
                    style={
                        "width": "250px",
                        "padding": "20px",
                        "backgroundColor": "#f8f9fa",
                        "borderRight": "1px solid #dee2e6",
                    },
                    children=[
                        html.H4("File Upload", style={"marginTop": "0"}),
                        html.H5("Experimental Data"),
                        dcc.Upload(
                            id="upload-experimental-data",
                            children=DEFAULT_UPLOAD_CHILDREN,
                            style={
                                "width": "100%",
                                "height": "60px",
                                "lineHeight": "60px",
                                "borderWidth": "1px",
                                "borderStyle": "dashed",
                                "borderRadius": "5px",
                                "textAlign": "center",
                                "margin": "10px 0",
                            },
                            multiple=False,
                        ),
                        html.Br(),
                        html.H5("CREASE Results"),
                        dcc.Upload(
                            id="upload-crease-results",
                            children=DEFAULT_UPLOAD_CHILDREN,
                            style={
                                "width": "100%",
                                "height": "60px",
                                "lineHeight": "60px",
                                "borderWidth": "1px",
                                "borderStyle": "dashed",
                                "borderRadius": "5px",
                                "textAlign": "center",
                                "margin": "10px 0",
                            },
                            multiple=False,
                        ),
                    ],
                ),
                html.Div(
                    id="main-content",
                    style={"flex": "1", "padding": "20px", "overflowY": "auto"},
                    children=[
                        dcc.Tabs(
                            id="tabs-main",
                            value="tab-scatter",
                            children=[
                                dcc.Tab(label="Scattering Fit", value="tab-scatter"),
                                dcc.Tab(
                                    label="Parameter Distributions",
                                    value="tab-distributions",
                                ),
                                dcc.Tab(label="Summary", value="tab-summary"),
                                dcc.Tab(
                                    label="Real-Space Reconstruction",
                                    value="tab-reconstruction",
                                ),
                            ],
                        ),
                        html.Div(id="tab-content-display", style={"marginTop": "20px"}),
                    ],
                ),
            ],
        ),
    ],
)

register_load_data_callbacks(app)
register_plotting_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True)
