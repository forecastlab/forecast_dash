import json

import dash_bootstrap_components as dbc
from dash import dcc, html
import dash

from common import breadcrumb_layout

dash.register_page(
    __name__, 
    title="About"
)

### load contributors, reasearch team
active_contributors = json.load(open("static_files/profiles.json"))[
    "active_contributors"
]

research_team = json.load(open("static_files/profiles.json"))[
    "research_team"
]

past_contributors = json.load(open("static_files/profiles.json"))[
    "past_contributors"
]

### functions for different part of about us page
def _about():
    return dbc.Col(
        [
            html.H1("About"),
            html.P(
                "This website aims to make business forecasting accessible to the wider community. Here you can find predictions of commonly used time series and the uncertainty around them."
            ),
            html.P(
                "The website implements classical forecasting models as well as the novel models developed by the members of the Time Series and Forecasting (TSF) research group in the University of Sydney Business School."
            ),
            html.P(
                [
                    """
The forecasts are updated daily and include the latest publicly available information. It is an open-source project, with code
                    """,
                    html.A(
                        "available here",
                        href="https://github.com/forecastlab/forecast_dash",
                    ),
                    ".",
                ]
            ),
        ],
        lg=6,
    )

def _mission():
    return dbc.Col(
        [
            html.Div(
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
            ),
        ],
        lg=6,
    )

def _builtby():
    return dbc.Col(
        [
            html.Div(
                dbc.Container(
                    dbc.Row(
                        dbc.Col(
                            [
                                html.H1(
                                    [
                                        "Built by the ",
                                        html.A(
                                            "Discipline of Business Analytics",
                                            href="https://www.sydney.edu.au/business/our-research/research-areas/business-analytics.html",
                                        ),
                                        " at the University of Sydney",
                                    ],
                                    className="display-6",
                                ),
                                html.A(
                                    [
                                        html.Img(
                                            src="/assets/USYD_logo.png"
                                        )
                                    ],
                                    href="https://www.sydney.edu.au/business/",
                                ),
                            ],
                            className="text-center",
                        ),
                    ),
                    className="px-4",
                ),
                className="rounded-3 py-5 mb-4",
            ),
        ],
        lg=12,
    )

def _section_title(title):
    return dbc.Col(
        html.H2(
            title,
            style={"margin-bottom": "32px"},
        ),
        lg=12,
    )

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
            className="text-center mb-5",
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

def _disclaimer():
    return dbc.Col(
        [
            html.P(
                "The University strives to keep information stored on this server up to date, but does not guarantee the accuracy, reliability or currency of the information. Any errors in the information that are brought to the University’s attention will be corrected as soon as possible. The University reserves the right to change at any time without notice any information stored on this server. This includes information about courses or units of study offered by the University."
            ),
            html.P(
                "The University of Sydney provides links to a number of external websites via this website. Monitoring and reviewing content of these third party external websites is not the responsibility of the University of Sydney nor does the University of Sydney endorse, approve or recommend the content, owners or operators of websites and applications available through this website."
            ),
            html.P(
                "The University accepts no liability for any loss or damage a person suffers because that person has directly or indirectly relied on any information stored on this server."
            ),
        ]
    )

def _contactus():
    return dbc.Col(
        [
            html.P(
                [
                    "Questions or suggestions? Feel free to reach out to the team by emailing: ",
                    html.A(
                        "app.forecasting-lab@sydney.edu.au",
                        href="mailto:app.forecasting-lab@sydney.edu.au",
                    ),
                ]
            ),
            html.P(
                "We will endeavour to address your query as soon as possible. "
            ),
        ]
    )

def body_layout():
    return dbc.Container(
        [
            breadcrumb_layout(
                [("Home", "/"), ("About", "")]
            ),
            dbc.Row([
                _about(),
                _mission(),
            ]),
            dbc.Row([_builtby()]),
            dbc.Row(_section_title("Active Contributors")),
            dbc.Row(parse_people(active_contributors)),
            dbc.Row(_section_title("Research Group")),
            dbc.Row(parse_people(research_team)),
            dbc.Row(_section_title("Past Contributors")),
            dbc.Row(parse_people(past_contributors)),
            dbc.Row(_section_title("Powered By")),
            dbc.Row(
                parse_poweredby("static_files/poweredby.json"),
                className="mb-5",
            ),
            dbc.Row(_section_title("Disclaimer")),
            dbc.Row(_disclaimer()),
            dbc.Row(_section_title("Contact us")),
            dbc.Row(_contactus()),
        ],
        style={"margin-bottom": "64px"},
        className="mb-5",
    )

layout = html.Div([
    dcc.Location(id="url", refresh=False),
    body_layout()
])