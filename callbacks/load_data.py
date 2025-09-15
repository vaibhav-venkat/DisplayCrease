from dash import Input, Output, State
from helpers.parsers import parse_contents


def register_callbacks(app):
    @app.callback(
        Output("exp-data", "data"),  # Output 1: The data goes into the store
        Output(
            "exp-data-filename", "children"
        ),  # Output 2: The filename goes into the feedback Div
        Input("upload-exp-data", "contents"),  # Input: Triggered by new file content...
        State("upload-exp-data", "filename"),  # ...and we also get the filename
    )
    def update_experimental_data(contents, filename):
        if contents is not None:
            data, feedback_message = parse_contents(contents, filename)
            return data, feedback_message
        else:
            return None, "No file loaded."

    @app.callback(
        Output("crease-results", "data"),
        Output("crease-results-filename", "children"),
        Input("upload-crease-results", "contents"),
        State("upload-crease-results", "filename"),
    )
    def update_crease_data(contents, filename):
        if contents is not None:
            data, feedback_message = parse_contents(contents, filename)
            return data, feedback_message
        else:
            return None, "No file loaded."
