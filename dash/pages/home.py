### dash related
from dash import html, dcc, callback, Output, Input
import dash
import dash_bootstrap_components as dbc

from urllib.parse import urlencode

import json
import pandas as pd

### utils
from common import (
    get_thumbnail_figure,
    get_forecast_data,
    component_news_4col,
    component_figs_2col,
    world_map_of_forecasts,
)

dash.register_page(__name__, path="/", title="Business Forecast Lab")

# load data source
with open("../shared_config/data_sources.json") as data_sources_json_file:
    series_list = json.load(data_sources_json_file)


### functions for different part of layout
# section 1: mission statement
def mission_statement():
    return html.Div(
        dbc.Container(
            dbc.Row(
                dbc.Col(
                    [
                        html.H1(
                            "Our Mission",
                            className="display-4",
                        ),
                        html.Hr(),
                        html.P(
                            html.Ul(
                                [
                                    html.Li(
                                        "To make forecasting models accessible to everyone.",
                                        className="lead",
                                    ),
                                    html.Li(
                                        "To provide the latest economic and financial forecasts of commonly used time series.",
                                        className="lead",
                                    ),
                                ],
                            ),
                        ),
                    ]
                ),
            ),
            className="px-4",
        ),
        className="bg-light rounded-3 py-5 mb-4",
    )


# section 2: main body
def _featured_latest_news():
    return [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3(
                            "Series Map",
                            style={"text-align": "center"},
                        ),
                        html.H6(
                            [
                                "Click on a country to view more. ",
                                html.A(
                                    [
                                        "Or search by name, country, tags and more!"
                                    ],
                                    href="/search/",
                                    className="text-decoration-none",
                                ),
                            ],
                            style={"text-align": "center"},
                            className="mb-0",
                        ),
                        html.Div(
                            dcc.Graph(
                                figure=world_map_of_forecasts(),
                                config={"displayModeBar": False},
                                id="choropleth",
                                # responsive= True,
                            ),
                            id="myDiv",
                        ),
                        html.Div(id="LinkOutCountry")
                        # ],
                        # href="/search/",
                        # ),
                    ],
                    lg=8,
                ),
                component_news_4col(),
            ],
            className="mb-5",
        ),
    ]


@callback(Output("home_url", "href"), [Input("choropleth", "clickData")])
def demo(clickData):
    if clickData:
        country = clickData["points"][0]["customdata"][0]
        return f"/search/?name={country}"


def _leaderboard():
    return dbc.Row(
        [
            dbc.Col(
                [
                    html.Div(
                        dbc.Container(
                            dbc.Row(
                                dbc.Col(
                                    [
                                        html.A(
                                            [
                                                html.H1(
                                                    "Leaderboard",
                                                    className="display-4",
                                                ),
                                            ],
                                            href="/leaderboard/",
                                            className="text-decoration-none text-reset",
                                        ),
                                        html.H2(
                                            "We backtest every model on every series.",
                                            className="mb-3",
                                        ),
                                        html.H2(
                                            "Daily.",
                                            className="mb-3",
                                        ),
                                        html.A(
                                            html.H4("Go to Leaderboard"),
                                            href="/leaderboard/",
                                            className="text-decoration-none",
                                        ),
                                    ],
                                    className="text-center",
                                ),
                            ),
                            className="px-4",
                        ),
                        className="bg-light rounded-3 py-5 mb-4",
                    ),
                ],
                lg=12,
            )
        ]
    )


def _link_search():
    return dbc.Row(
        [
            dbc.Col(
                [
                    html.Div(
                        dbc.Container(
                            dbc.Row(
                                dbc.Col(
                                    [
                                        html.H2(
                                            "Looking for Something?",
                                            className="mb-3",
                                        ),
                                        html.A(
                                            html.H4(
                                                "Filter by name, country, tags and more!",
                                                className="mb-3",
                                            ),
                                            href="/search/",
                                            className="text-decoration-none",
                                        ),
                                    ],
                                    className="text-center",
                                ),
                            ),
                            className="px-4",
                        ),
                        className="bg-light rounded-3 py-5 mb-4",
                    ),
                ],
                lg=12,
            )
        ]
    )


def snapshot(series_name, callout_text, link_text, link_url, direction="left"):
    
    text_col = dbc.Col(
        [
            html.H2(
                callout_text
            ),
            html.A(
                html.H4(link_text), #html.H4("View all Australian forecasts"),
                href=link_url,#f"/search?{urlencode({'name': 'Australia'})}",
                className="text-decoration-none",
            ),
        ],
        lg=4,
        className="text-center align-self-center",
    )

    fig_col = dbc.Col(
        [
            html.A(
                [
                    dcc.Graph(
                        figure=get_thumbnail_figure(
                            get_forecast_data(series_name),
                            lg=8,
                        ),
                        config={"displayModeBar": False},
                    )
                ],
                href=f"/series?{urlencode({'title': series_name})}",
            ),
        ],
        lg=8,
    )

    if direction == "left":
        return [
            dbc.Row(
                [
                    text_col,
                    fig_col

                ],
                className="d-flex",
            )
        ]

    else:

        return [
            dbc.Row(
                [
                    fig_col,
                    text_col,

                ],
                className="d-flex",
            )
        ]

def layout():
    return html.Div(
        [
            dcc.Location(id="home_url", refresh=True),
            mission_statement(),
            dbc.Container(
                [
                    *_featured_latest_news(),
                    _link_search(),
                    *snapshot(
                        "Australian Monthly Inflation (CPI)",
                        "AU" +" inflation has peaked".title(),
                        html.H4("View all Inflation forecasts"),
                        f"/search?{urlencode({'name': 'Inflation'})}",
                    ),
                    *snapshot(
                        "Australian GDP Growth",
                        "AU GDP " + "growth remains flat".title(),
                        html.H4("View all GDP forecasts"),
                        f"/search?{urlencode({'name': 'GDP'})}",
                        direction="right"
                    ),
                    _leaderboard(),
                    *snapshot(
                        "US Personal Consumption Expenditures: Chain-type Price Index (% Change, 1 Year)",
                        "US " + "inflation stabilised".title(),
                        html.H4("View all US forecasts"),
                        f"/search?{urlencode({'name': 'United States'})}",
                    ),
                    *snapshot(
                        "UK Inflation (RPI)",
                        "UK " + "inflation stabilises".title(),
                        html.H4("View all UK forecasts"),
                        f"/search?{urlencode({'name': 'United Kingdom'})}",
                        direction="right"
                    ),
                    *snapshot(
                        "Australian (Sydney) Change in House Prices",
                        "AU" +" House prices slow down".title(),
                        html.H4("View all Australian forecasts"),
                        f"/search?{urlencode({'name': 'Australia'})}",
                    )
                ]
            )
        ]
    )
