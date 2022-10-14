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
                            "World Map of Featured Series",
                            style={"text-align": "center"},
                        ),
                        # html.A(
                        #     [
                        html.Div(
                            dcc.Graph(
                                figure=world_map_of_forecasts(),
                                config={"displayModeBar": False},
                                id="choropleth",
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
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2(
                            "US Recovery from COVID-19",
                        ),
                        html.A(
                            html.H4("View all US forecasts"),
                            href="/search/?name=&tags=US",
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
                                        get_forecast_data("US Unemployment"),
                                        lg=8,
                                    ),
                                    config={"displayModeBar": False},
                                )
                            ],
                            href=f"/series?{urlencode({'title': 'US Unemployment'})}",
                        ),
                    ],
                    lg=8,
                ),
            ],
            className="d-flex",
        ),
    ]


@callback(
    Output("LinkOutCountry", "children"), [Input("choropleth", "clickData")]
)
def update_figure(clickData):
    countries = pd.read_csv("../data/CountriesList.csv")

    if clickData is not None:
        location = clickData["points"][0]["location"]
        selection = countries["Country"][countries["Code"] == location].values[
            0
        ]

        # if location not in selections:
        #     selections.add(location)
        # else:
        #     selections.remove(location)

        return html.A(
            [f"Check out the forecast series in the {selection}"],
            href=f"search/?name={selection}",
        )
    else:
        return ""


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
    return component_figs_2col(
        "UK Snapshot",
        [
            "UK Unemployment",
            "UK Inflation (RPI)",
        ],
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


def main_body(feature_series_title):
    return dbc.Container(
        [
            *_featured_latest_news(feature_series_title),
            _leaderboard(),
            _uk_snapshot(),
            _link_search(),
        ]
    )


### The layout variable
feature_series_title = "Australian Inflation (CPI)"  # can remove this

layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        mission_statement(),
        main_body(feature_series_title),
    ]
)
