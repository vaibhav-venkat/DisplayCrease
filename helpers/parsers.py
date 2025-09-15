import base64
import io
import pandas as pd
from dash import html


def parse_contents(contents, filename):
    """
    Parses the content of an uploaded file.

    Args:
        contents (str): The base64 encoded string of the file content.
        filename (str): The name of the uploaded file.

    Returns:
        tuple: A tuple containing the parsed DataFrame and an Div.
    """
    content_type, content_string = contents.split(",")

    decoded = base64.b64decode(content_string)
    try:
        if "csv" in filename:
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        elif "txt" in filename or "dat" in filename:
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), sep=r"\s+")
        else:
            return None, html.Div(["Invalid file type. Please upload a CSV or TXT ."])

    except Exception as e:
        print(e)
        return None, html.Div(["There was an error processing this file."])

    # Data is stored as JSON in dcc.Store
    # The feedback message shows the filename.
    return df.to_json(date_format="iso", orient="split"), html.Div(
        f"Loaded: {filename}"
    )
