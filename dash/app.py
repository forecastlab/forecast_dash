from flask import Flask
from dash import Dash, html, callback
import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from common import header, footer


server = Flask(__name__)
app = Dash(
    __name__,
    server=server,
    url_base_pathname="/",
    use_pages=True,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://use.fontawesome.com/releases/v5.8.1/css/all.css",
    ],
)

app.layout = html.Div([header(), dash.page_container, footer(),])

### callback for toggling the collapse on small screens
@callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


if __name__ == "__main__":
    server.run(host="0.0.0.0", port=80, debug=True)
