import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

from common import MarkdownApp, BootstrapApp, header, breadcrumb_layout, footer

import json


class Methodology(MarkdownApp):

    markdown = """
# Methodology

**This page is under construction.**

It will contain the description of the models and other aspects of the
methodology used to forecast the time series.

While we are busy with this document, we recommend “Forecasting: Principles
and Practice” textbook freely available at
[otexts.com/fpp2/](https://otexts.com/fpp2/)
    """


def parse_people(filepath):

    with open(filepath) as person_file:
        person_list = json.load(person_file)

        result = []

        for person_dict in person_list:
            result.extend(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Img(
                                        src=person_dict["image_url"],
                                        height="200px",
                                    )
                                ],
                                lg=2,
                                style={"margin-top": "16px"},
                            ),
                            dbc.Col(
                                [
                                    html.H4(
                                        person_dict["name"], className="mt-3"
                                    ),
                                    html.P(
                                        person_dict["bio"],
                                        style={"whiteSpace": "pre-wrap"},
                                    ),
                                    html.A(
                                        person_dict["homepage"],
                                        href=person_dict["homepage"],
                                    ),
                                ],
                                lg=9,
                            ),
                        ],
                        style={"margin-bottom": "16px"},
                    )
                ]
            )

    return result


class About(BootstrapApp):
    def setup(self):

        self.layout = html.Div(
            header()
            + [
                dcc.Location(id="url", refresh=False),
                dbc.Container(
                    [
                        breadcrumb_layout(
                            [("Home", "/"), (f"{self.title}", "")]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.H2("Our Mission"),
                                        html.Ol(
                                            [
                                                html.Li(
                                                    "To make forecasting models accessible to everyone."
                                                ),
                                                html.Li(
                                                    "To provide the latest financial and economic forecasts of the commonly used time series."
                                                ),
                                            ]
                                        ),
                                    ],
                                    lg=12,
                                )
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.H2("About"),
                                        html.P(
                                            "The Business Forecast Lab was established in ...."
                                        ),
                                    ],
                                    lg=12,
                                )
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [html.H2("Code Contributors")]
                                    + parse_people(
                                        "static_files/contributors.json"
                                    ),
                                    lg=12,
                                )
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [html.H2("Research Group Leaders")]
                                    + parse_people(
                                        "static_files/research_team.json"
                                    ),
                                    lg=12,
                                )
                            ]
                        ),
                    ]
                    + footer(),
                    style={"margin-bottom": "64px"},
                    className="mb-5",
                ),
            ]
        )
