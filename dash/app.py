from flask import Flask
from dash import Dash, html, dcc
import dash
import dash_bootstrap_components as dbc
# from multipage import Route, MultiPageApp
# from pages import Index, Series, Search, Leaderboard
# from pages_static import Methodology, About

# from blog import BlogSection
from common import header, footer


server = Flask(__name__)
app = Dash(
    __name__,
    server=server,
    url_base_pathname="/",
    use_pages=True,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        'https://use.fontawesome.com/releases/v5.8.1/css/all.css'
    ],
)

app.layout = html.Div([
    *header(),
    dash.page_container,
    *footer(), # footer has two elements
])

if __name__ == "__main__":
    server.run(host="0.0.0.0", port=8001, debug=True)
