import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

from common import MarkdownApp, BootstrapApp, header, breadcrumb_layout, footer

import json


class Methodology(MarkdownApp):

    markdown = """
# Data

Data are sourced from:
- [Australian Macro Database](http://ausmacrodata.org)
- [St Louis Federal Reserve](https://api.stlouisfed.org)
- [UK Office of National Statistics](https://www.ons.gov.uk)

# Methodology

The available models are listed in the [Leaderboard](/leaderboard/).
These are based on the benchmark models used in the M4 Competition \[0\].

The models are run on each dataset according to the time series cross-validation
scheme described in \[1\], Sect 3.4, with forecast horizons of length 1-8.

![time series cross-validation](https://otexts.com/fpp2/fpp_files/figure-html/cv1-1.png)  
\(Image reproduced from \[1\] with permission.\)

The forecast accuracy or cross-validation score is computed by averaging 
the mean-squared forecast error over the test sets and forecast horizons. 
The model with the best forecast accuracy is selected by the Forecast Lab 
as the preferred model. Forecasts from the other available models may be 
selected from the drop-down menu in each Series page.


\[0\] Makridakis, S., Spiliotis, E. and Assimakopoulos, V.,  
      _The M4 Competition: 100,000 time series and 61 forecasting methods,_  
      Int. J. Forecasting 36 \(2020\) 54-74

\[1\] Hyndman, R. and Athanasopoulos, G.,  
      _Forecasting: Principles and Practice_
      OTexts: Melbourne, Australia.  
      [otexts.com/fpp2/](https://otexts.com/fpp2/)

    """


def parse_people(person_list):

    icon_map = {
        "home": "fas fa-home",
        "github": "fab fa-github",
        "work": "fas fa-building",
        "university": "fa fa-university",
    }

    return [
        dbc.Col(
            [
                html.Img(
                    src=person_dict["image_url"],
                    height="148px",
                    className="rounded-circle shadow",
                ),
                html.H5(
                    person_dict["name"],
                    className="mt-4 font-weight-medium mb-0",
                ),
                html.H6(
                    person_dict["affiliation"],
                    className="subtitle mb-3 text-muted",
                ),
            ]
            + (
                [
                    html.Ul(
                        [
                            html.Li(
                                html.A(
                                    html.I(
                                        className=f"{icon_map[link_type]} fa-lg mr-3"
                                    ),
                                    href=link_value,
                                ),
                                className="list-inline-item",
                            )
                            for link_type, link_value in person_dict[
                                "links"
                            ].items()
                        ],
                        className="list-inline",
                    )
                ]
                if "links" in person_dict
                else []
            )
            + [
                html.P(person_dict["bio"], className="text-justify"),
            ],
            lg=4,
            sm=6,
            className="text-center",
        )
        for person_dict in person_list
    ]


def parse_poweredby(filepath):

    with open(filepath) as poweredby_file:
        poweredby_list = json.load(poweredby_file)

        return [
            dbc.Col(
                [
                    html.A(
                        [
                            html.Img(
                                src=poweredby_dict["image_url"],
                                height="96px",
                                style={"margin-bottom": "8px"},
                            ),
                            html.H5(poweredby_dict["name"]),
                        ],
                        href=poweredby_dict["url"],
                    )
                ],
                lg=2,
                md=3,
                xs=6,
                className="text-center",
                style={"margin-bottom": "16px"},
            )
            for poweredby_dict in poweredby_list
        ]


class About(BootstrapApp):
    def setup(self):

        contributors = json.load(open("static_files/profiles.json"))[
            "contributors"
        ]

        research_team = json.load(open("static_files/profiles.json"))[
            "research_team"
        ]

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
                                        html.H2("About"),
                                        html.P(
                                            [
                                                """
This website is an intuitive tool that makes business forecasting accessible to the wider community. You can easily obtain predictions of commonly used variables together with the uncertainty around them. The website implements classical forecasting models as well as the novel models and methods developed by the members of the Time Series and Forecasting (TSF) research group in the University of Sydney Business School. The website visualizes and summarizes the forecasting results in an easy-to-understand manner. The forecasts are updated daily and include the latest publicly available information. It is an open-source project under the AGPL license, see 
                                                """,
                                                html.A(
                                                    "https://github.com/forecastlab/forecast_dash",
                                                    href="https://github.com/forecastlab/forecast_dash",
                                                ),
                                                " .",
                                            ]
                                        ),
                                    ],
                                    lg=6,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Jumbotron(
                                            [
                                                html.H1("Our Mission"),
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
                                            ]
                                        )
                                    ],
                                    lg=6,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.H2(
                                        "Built by the University of Sydney\n Discipline of Business Analytics"
                                    ),
                                    width=6,
                                    align="center",
                                ),
                                dbc.Col(
                                    html.A(
                                        [html.Img(src="assets/USYD_logo.png")],
                                        href="https://www.sydney.edu.au/business/",
                                    ),
                                    width=6,
                                ),
                            ],
                            justify="between",
                        ),
                        dbc.Row(
                            dbc.Col(
                                html.H2(
                                    "Powered By",
                                    style={"margin-bottom": "32px"},
                                ),
                                lg=12,
                            )
                        ),
                        dbc.Row(
                            parse_poweredby("static_files/poweredby.json")
                        ),
                        dbc.Row(
                            dbc.Col(
                                html.H2(
                                    "Core Contributors",
                                    style={"margin-bottom": "32px"},
                                ),
                                lg=12,
                            )
                        ),
                        dbc.Row(parse_people(contributors)),
                        dbc.Row(
                            dbc.Col(
                                html.H2(
                                    "Research Group Leaders",
                                    style={"margin-bottom": "32px"},
                                ),
                                lg=12,
                            )
                        ),
                        dbc.Row(parse_people(research_team)),
                    ]
                    + footer(),
                    style={"margin-bottom": "64px"},
                    className="mb-5",
                ),
            ]
        )
