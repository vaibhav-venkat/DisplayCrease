import base64
import io
import pandas as pd
from dash import html


def parse_contents(contents, filename):
    """
    Parses the content of an uploaded file into a JSON string for dcc.Store.
    """
    if contents is None:
        return None, "No file loaded."

    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    try:
        if "csv" in filename:
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        elif "txt" in filename or "dat" in filename:
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), sep=r"\s+")
        else:
            return None, html.Div(
                ["Invalid file type. Please upload a CSV, TXT, or DAT file."]
            )

    except Exception as e:
        print(f"Error parsing file {filename}: {e}")
        return None, html.Div([f"There was an error processing {filename}."])

    return df.to_json(date_format="iso", orient="split"), html.Div(
        f"Loaded: {filename}"
    )
