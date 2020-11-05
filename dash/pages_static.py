import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

from common import MarkdownApp, BootstrapApp, header, breadcrumb_layout, footer

import json


class Methodology(MarkdownApp):

    markdown = """
# Data

Data are sourced from the following sources:
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
                                style={'margin-bottom': '8px'}
                            ),
                            html.H5(poweredby_dict["name"])
                        ],
                        href=poweredby_dict["url"]
                    )
                ],
                lg=2, md=3, xs=6,
                className='text-center',
                style={'margin-bottom': '16px'}
            )

            for poweredby_dict in poweredby_list
        ]

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
                                        html.H2("About"),
                                        html.P(
                                            """
                                            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nam ultricies ante
                                            turpis, eget lobortis erat feugiat quis. Duis sed rutrum sapien, ut ullamcorper
                                            turpis. Praesent ut nunc lobortis lorem gravida bibendum. Aenean elementum dapibus
                                            felis vitae posuere.Nulla semper erat vitae sollicitudin elementum.
                                            """
                                        ),
                                        html.P(
                                            """
                                            Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac
                                            turpis egestas. Praesent semper fermentum erat ut cursus.Etiam gravida dui justo,
                                            at sodales ante euismod ac. Maecenas porta nisi ut lacus vehicula imperdiet. Sed
                                            facilisis dui id orci volutpat, id porta est imperdiet.
                                            """
                                        ),
                                    ],
                                    lg=6,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Jumbotron([
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
                                        ])

                                    ],
                                    lg=6,
                                )
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(html.H2("Powered By", className='text-center', style={'margin-bottom': '32px'}), lg=12,
                                )
                            ]
                        ),
                        dbc.Row(
                                parse_poweredby("static_files/poweredby.json"),
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
