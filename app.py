import dash
from dash import dcc, html, dash_table

from callbacks.load_data import register_callbacks

app = dash.Dash(__name__)


app.layout = html.Div(
    id="main-container",
    style={"fontFamily": "sans-serif"},
    children=[
        dcc.Store(id="exp-data"),
        dcc.Store(id="crease-results"),
        html.H1(
            "DisplayCrease",
            style={"textAlign": "center", "margin": "20px"},
        ),
        html.Div(
            id="content-container",
            style={"display": "flex"},
            children=[
                # --- Sidebar for Controls (Left Column) ---
                html.Div(
                    id="sidebar",
                    style={
                        "width": "250px",
                        "padding": "20px",
                        "backgroundColor": "#f8f9fa",
                        "borderRight": "1px solid #dee2e6",
                    },
                    children=[
                        html.H4("Data", style={"marginTop": "0"}),
                        # Upload component for experimental scattering data
                        dcc.Upload(
                            id="upload-exp-data",
                            children=html.Div(
                                [
                                    "Upload Data",
                                ]
                            ),
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
                        html.Div(id="exp-data-filename"),
                        html.Br(),
                        # Upload component for CREASE results
                        dcc.Upload(
                            id="upload-crease-results",
                            children=html.Div(["Load CREASE results"]),
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
                        html.Div(id="crease-results-filename"),
                    ],
                ),  # End of Sidebar
                # --- Main Content Area (Right Column) ---
                html.Div(
                    id="main-content",
                    style={"flex": "1", "padding": "20px"},
                    children=[
                        # Tabs for different views
                        dcc.Tabs(
                            id="tabs-main",
                            value="tab-scatter",
                            children=[
                                dcc.Tab(label="Scattering Fit", value="tab-scatter"),
                                dcc.Tab(
                                    label="Real-Space Reconstruction",
                                    value="tab-reconstruction",
                                ),
                                dcc.Tab(
                                    label="Parameter Distributions",
                                    value="tab-distributions",
                                ),
                                dcc.Tab(label="Summary", value="tab-summary"),
                            ],
                        ),
                        # Div to hold the content for the selected tab
                        # In a functional app, a callback would update this Div's children.
                        # For this bare layout, we will just show all components.
                        html.Div(
                            id="tab-content-display",
                            style={"marginTop": "20px"},
                            children=[
                                # Placeholder for the main graph (Scattering plot, 3D structure, etc.)
                                dcc.Graph(
                                    id="main-graph",
                                    figure={
                                        "layout": {
                                            "title": "Plot Area (e.g., Scattering Fit)",
                                            "xaxis": {"title": "q"},
                                            "yaxis": {"title": "Intensity"},
                                        }
                                    },
                                    style={"height": "500px"},
                                ),
                                html.Hr(style={"margin": "20px 0"}),
                                # Placeholder for the summary table
                                html.H4("Summary Metrics"),
                                dash_table.DataTable(
                                    id="summary-table",
                                    columns=[
                                        {"name": "Metric", "id": "metric"},
                                        {"name": "Value", "id": "value"},
                                    ],
                                    data=[
                                        {"metric": "Example Metric 1", "value": "..."},
                                        {"metric": "Example Metric 2", "value": "..."},
                                    ],
                                    style_cell={"textAlign": "left"},
                                    style_header={
                                        "backgroundColor": "rgb(230, 230, 230)",
                                        "fontWeight": "bold",
                                    },
                                ),
                            ],
                        ),  # End of Tab Content
                    ],
                ),  # End of Main Content
            ],
        ),  # End of Content Container
    ],
)  # End of App Layout
register_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True)
