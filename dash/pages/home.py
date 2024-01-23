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
def _featured_latest_news(feature_series_title):
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
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2(
                            "US "
                            + "inflation shows strong signs of stabilisation".title(),
                        ),
                        html.A(
                            html.H4("View all US forecasts"),
                            href=f"/search?{urlencode({'name': 'United States'})}",
                            className="text-decoration-none",
                        ),
                    ],
                    lg=4,
                    className="text-center align-self-center",
                ),
                dbc.Col(
                    [
                        html.A(
                            [
                                dcc.Graph(
                                    figure=get_thumbnail_figure(
                                        get_forecast_data(
                                            "US Personal Consumption Expenditures: Chain-type Price Index (% Change, 1 Year)"
                                        ),
                                        lg=8,
                                    ),
                                    config={"displayModeBar": False},
                                )
                            ],
                            href=f"/series?{urlencode({'title': 'US Personal Consumption Expenditures: Chain-type Price Index (% Change, 1 Year)'})}",
                        ),
                    ],
                    lg=8,
                ),
            ],
            className="d-flex",
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


def _uk_snapshot():
    return [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2(
                            "UK " + "struggles to contain inflation".title(),
                        ),
                        html.A(
                            html.H4("View all UK forecasts"),
                            href=f"/search?{urlencode({'name': 'United Kingdom'})}",
                            className="text-decoration-none",
                        ),
                    ],
                    lg=4,
                    className="text-center align-self-center",
                ),
                dbc.Col(
                    [
                        html.A(
                            [
                                dcc.Graph(
                                    figure=get_thumbnail_figure(
                                        get_forecast_data(
                                            "UK Inflation (RPI)"
                                        ),
                                        lg=8,
                                    ),
                                    config={"displayModeBar": False},
                                )
                            ],
                            href=f"/series?{urlencode({'title': 'UK Inflation (RPI)'})}",
                        ),
                    ],
                    lg=8,
                ),
            ],
            className="d-flex",
        )
    ]


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


def _au_snapshot():
    return [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2(
                            "Have "
                            + "AU"
                            + " house prices reached the bottom?".title()
                        ),
                        html.A(
                            html.H4("View all Australian forecasts"),
                            href=f"/search?{urlencode({'name': 'Australia'})}",
                            className="text-decoration-none",
                        ),
                    ],
                    lg=4,
                    className="text-center align-self-center",
                ),
                dbc.Col(
                    [
                        html.A(
                            [
                                dcc.Graph(
                                    figure=get_thumbnail_figure(
                                        get_forecast_data(
                                            "Australian (Sydney) Change in House Prices"
                                        ),
                                        lg=8,
                                    ),
                                    config={"displayModeBar": False},
                                )
                            ],
                            href=f"/series?{urlencode({'title': 'Australian (Sydney) Change in House Prices'})}",
                        ),
                    ],
                    lg=8,
                ),
            ],
            className="d-flex",
        )
    ]


def main_body(feature_series_title):
    return dbc.Container(
        [
            *_featured_latest_news(feature_series_title),
            _leaderboard(),
            *_uk_snapshot(),
            _link_search(),
            *_au_snapshot(),
        ]
    )


### The layout variable
feature_series_title = "Australian Inflation (CPI)"  # can remove this

layout = html.Div(
    [
        dcc.Location(id="home_url", refresh=True),
        mission_statement(),
        main_body(feature_series_title),
    ]
)
