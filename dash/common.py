import os
from abc import ABC, abstractmethod

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from datetime import datetime
import humanize
import pytz


def header():
    from app import nav_routes

    return [
        dbc.Navbar(
            dbc.Container(
                [
                    dbc.Row(
                        [
                            # dbc.Col(html.Img(src="/assets/USYD_coa_reversed.png", height="30px")),
                            dbc.Col(
                                html.A(
                                    [
                                        html.Img(
                                            src="/assets/USYD_coa_reversed.png",
                                            height="30px",
                                        )
                                    ],
                                    href="https://www.sydney.edu.au/business/",
                                )
                            ),
                            dbc.Col(
                                dbc.NavbarBrand(
                                    "Forecast Lab", className="ms-2", href="/"
                                )
                            ),
                        ],
                        align="center",
                        className="g-0",
                    ),
                    dbc.Col(
                        dbc.Row(
                            [
                                dbc.NavbarToggler(id="navbar-toggler"),
                                dbc.Collapse(
                                    dbc.Nav(
                                        [
                                            dbc.NavItem(
                                                dbc.NavLink(
                                                    x[1],
                                                    href=x[2],
                                                    external_link=True,
                                                )
                                            )
                                            for x in nav_routes
                                        ]
                                        + [
                                            dbc.NavItem(
                                                dbc.NavLink(
                                                    html.I(
                                                        className="fab fa-github fa-lg"
                                                    ),
                                                    href="https://github.com/forecastlab/forecast_dash",
                                                    external_link=True,
                                                )
                                            )
                                        ],
                                        # make sure nav takes up the full width for auto
                                        # margin to get applied
                                        className="w-100",
                                    ),
                                    id="navbar-collapse",
                                    is_open=False,
                                    navbar=True,
                                ),
                            ],
                            # the row should expand to fill the available horizontal space
                            className="flex-grow-1",
                        ),  # close row
                        lg="expand",
                    ),  # close col
                ],
            ),  # close containter
            color="dark",
            dark=True,
        )
    ]


def breadcrumb_layout(crumbs):

    return dbc.Nav(
        [
            html.Ol(
                [
                    html.Li(
                        html.A(crumb[0], href=crumb[1]),
                        className="breadcrumb-item",
                    )
                    for crumb in crumbs[:-1]
                ]
                + [
                    html.Li(
                        crumbs[-1][0],
                        id="breadcrumb",
                        className="breadcrumb-item active",
                    )
                ],
                className="breadcrumb",
            )
        ],
        navbar=True,
        class_name="bg-light rounded-3 px-3 pt-3 mb-3",
    )


def component_git_version():

    git_hash = ""
    git_shorthash = "Unknown commit"
    git_time = "00:00"
    git_author = "Unknown author"
    git_subject = ""

    # Get the current git status
    # %n: literal newline
    # %H: commit hash
    # %h: abbreviated commit hash
    # %ai: author date
    # %an: author name
    # %s: subject
    # Gotcha: The seemingly redundant --git-dir ../.git is a workaround for
    # docker container isolation: .git is exported to /.git as read-only volume.
    git_cmd = (
        'git --git-dir ../.git show --no-patch --format="%H%n%h%n%ai%n%an%n%s"'
    )
    git_output = os.popen(git_cmd).read().splitlines()

    if len(git_output) >= 5:
        git_hash = git_output[0]
        git_shorthash = git_output[1]
        git_time = git_output[2]

        natural_time = humanize.naturaltime(
            datetime.strptime(git_time, "%Y-%m-%d %H:%M:%S %z"),
            when=datetime.now(tz=pytz.timezone("Australia/Sydney")),
        )
    else:
        natural_time = "Error Loading Timestamp"

    github_home_url = "https://github.com/forecastlab/forecast_dash/"
    github_patch_url = github_home_url + "commit/" + git_hash

    return dbc.Col(
        [
            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            html.H5(
                                [
                                    "Version ",
                                    html.A(
                                        git_shorthash,
                                        href=github_patch_url,
                                        className="text-decoration-none",
                                    ),
                                ],
                                className="card-title",
                            ),
                            html.H6(
                                f"Committed {natural_time}",
                                className="card-subtitle mb-2 text-muted",
                            ),
                            html.P(
                                [
                                    html.A(
                                        "Development homepage",
                                        href=github_home_url,
                                        className="text-decoration-none",
                                    ),
                                    " on ",
                                    html.A(
                                        "GitHub",
                                        href="https://github.com/",
                                        className="text-decoration-none",
                                    ),
                                ]
                            ),
                        ]
                    )
                ],
                color="dark",
                outline=True,
            ),
        ]
    )


def footer():
    from app import home_route, nav_routes

    return [
        dbc.Row(
            dbc.Col(html.Hr(style={"margin-top": "64px"}), lg=12),
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Nav(
                            [
                                dbc.NavItem(
                                    dbc.NavLink(
                                        x[1], href=x[2], external_link=True
                                    )
                                )
                                for x in [home_route] + nav_routes
                            ],
                            vertical="md",
                        )
                    ],
                    lg=7,
                    style={"margin-bottom": "16px"},
                ),
                component_git_version(),
            ],
            style={"margin-bottom": "64px"},
        ),
    ]


class BootstrapApp(dash.Dash, ABC):
    def __init__(self, name, server, url_base_pathname):

        external_scripts = [
            {
                "src": "https://kit.fontawesome.com/b4d76f3ee0.js",
                "crossorigin": "anonymous",
            }
        ]

        external_stylesheets = [dbc.themes.BOOTSTRAP]

        super().__init__(
            name=name,
            server=server,
            url_base_pathname=url_base_pathname,
            external_stylesheets=external_stylesheets,
            external_scripts=external_scripts,
            meta_tags=[
                {
                    "name": "viewport",
                    "content": "width=device-width, initial-scale=1",
                }
            ],
        )

        self.title = name

        self.setup()

    @abstractmethod
    def setup(self):
        pass


class MarkdownApp(BootstrapApp):
    @property
    @classmethod
    @abstractmethod
    def markdown(cls):
        return NotImplementedError

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
                        dcc.Markdown(type(self).markdown),
                    ]
                    + footer()
                ),
            ]
        )
