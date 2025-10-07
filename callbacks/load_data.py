from dash import Input, Output, State, html
from helpers.parsers import parse_contents

DEFAULT_UPLOAD_CHILDREN = html.Div(["Drag and Drop or ", html.A("Select File")])


def register_callbacks(app):
    """Registers all data loading callbacks."""

    @app.callback(
        Output("experimental-data-store", "data"),
        Output("upload-experimental-data", "children"),
        Input("upload-experimental-data", "contents"),
        State("upload-experimental-data", "filename"),
    )
    def update_experimental_data(contents, filename):
        if contents is not None:
            data, _ = parse_contents(contents, filename)
            if data is not None:
                return data, html.Div(f"Loaded: {filename}")
            else:
                return None, html.Div(
                    f"Error loading {filename}", style={"color": "red"}
                )

        return None, DEFAULT_UPLOAD_CHILDREN

    @app.callback(
        Output("crease-results-store", "data"),
        Output("upload-crease-results", "children"),  # NEW OUTPUT TARGET
        Input("upload-crease-results", "contents"),
        State("upload-crease-results", "filename"),
    )
    def update_crease_data(contents, filename):
        if contents is not None:
            data, _ = parse_contents(contents, filename)
            if data is not None:
                return data, html.Div(f"Loaded: {filename}")
            else:
                return None, html.Div(
                    f"Error loading {filename}", style={"color": "red"}
                )

        return None, DEFAULT_UPLOAD_CHILDREN
