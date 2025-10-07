import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dcc, html, dash_table


def register_callbacks(app):
    """Registers all callbacks related to plotting and table displays."""

    @app.callback(
        Output("tab-content-display", "children"), Input("tabs-main", "value")
    )
    def render_tab_content(tab):
        if tab == "tab-scatter":
            return dcc.Graph(id="scatter-plot-graph", style={"height": "550px"})
        elif tab == "tab-distributions":
            return dcc.Graph(id="violin-plot-graph", style={"height": "550px"})
        elif tab == "tab-summary":
            return dash_table.DataTable(
                id="summary-table",
                style_cell={"textAlign": "left"},
                style_header={
                    "backgroundColor": "rgb(230, 230, 230)",
                    "fontWeight": "bold",
                },
                page_size=10,
            )
        elif tab == "tab-reconstruction":
            return html.Div(
                "3D structure generation functionality will be added here.",
                style={"padding": "20px"},
            )
        return html.P("Select a tab")

    @app.callback(
        Output("scatter-plot-graph", "figure"), Input("experimental-data-store", "data")
    )
    def update_scatter_plot(jsonified_exp_data):
        if jsonified_exp_data is None:
            return go.Figure().update_layout(
                title="Upload experimental data to see the scattering plot"
            )

        df = pd.read_json(jsonified_exp_data, orient="split")

        try:
            df_pivot = df.pivot(index="qy", columns="qx", values="intensity")

            fig = go.Figure(
                data=go.Contour(
                    z=df_pivot.values,
                    x=df_pivot.columns,
                    y=df_pivot.index,
                    contours_coloring="heatmap",
                    line_width=0,
                    connectgaps=False,
                )
            )

            fig.update_layout(
                title="2D Scattering Profile",
                xaxis_title="qx",
                yaxis_title="qy",
                xaxis=dict(scaleanchor="y", scaleratio=1),
                yaxis=dict(autorange="reversed"),
            )

            return fig

        except Exception as e:
            print(f"Error creating 2D plot: {e}")
            return go.Figure().update_layout(
                title="Error: Could not pivot data. Ensure columns are 'qx', 'qy', 'intensity'."
            )

    @app.callback(
        Output("violin-plot-graph", "figure"), Input("crease-results-store", "data")
    )
    def update_violin_plot(jsonified_crease_data):
        if jsonified_crease_data is None:
            return go.Figure().update_layout(
                title="Upload CREASE results to see parameter distributions"
            )

        df = pd.read_json(jsonified_crease_data, orient="split")

        fig = go.Figure()

        primary_axis_params = ["D_A", "omega_deg", "alpha_deg"]
        secondary_axis_params = ["e_mu", "e_sigma", "kappa_log", "dh", "lh", "nx"]

        for col in primary_axis_params:
            if col in df.columns:
                fig.add_trace(
                    go.Violin(
                        y=df[col],
                        name=col,
                        box_visible=True,
                        meanline_visible=True,
                    )
                )

        for col in secondary_axis_params:
            if col in df.columns:
                fig.add_trace(
                    go.Violin(
                        y=df[col],
                        name=col,
                        box_visible=True,
                        meanline_visible=True,
                        yaxis="y2",
                    )
                )

        fig.update_layout(
            title_text="Distributions of Optimized Structural Features",
            showlegend=False,
            yaxis=dict(title="<b>Large Scale Parameters</b> (D, ω, α)"),
            yaxis2=dict(
                title="<b>Small Scale / Normalized Parameters</b>",
                overlaying="y",
                side="right",
            ),
        )

        return fig

    @app.callback(
        Output("summary-table", "data"),
        Output("summary-table", "columns"),
        Input("crease-results-store", "data"),
    )
    def update_summary_table(jsonified_crease_data):
        if jsonified_crease_data is None:
            return [], []

        df = pd.read_json(jsonified_crease_data, orient="split")

        summary_df = df.describe().reset_index()
        summary_df = summary_df.rename(columns={"index": "Statistic"})

        columns = [{"name": i, "id": i} for i in summary_df.columns]
        data = summary_df.to_dict("records")

        return data, columns
