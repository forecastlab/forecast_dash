import os
from abc import ABC, abstractmethod

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html


def header():
    from app import nav_routes

    return [
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink(x[1], href=x[2], external_link=True))
                for x in nav_routes
            ],
            brand="Forecast Lab",
            brand_href="/",
            brand_external_link=True,
            color="dark",
            dark=True,
            expand="lg",
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
    )


def component_git_version():

    # Get the current git status
    # %n: literal newline
    # %H: commit hash
    # %h: abbreviated commit hash
    # %ai: author date
    # %an: author name
    # %s: subject
    git_output = (
        os.popen('git show --no-patch --format="%H%n%h%n%ai%n%an%n%s"')
        .read()
        .splitlines()
    )

    git_hash = git_output[0]
    git_shorthash = git_output[1]
    git_time = git_output[2]
    git_author = git_output[3]

    # Gotcha: git_subject might contain newlines
    git_subject = "\n".join(git_output[4:])

    github_home_url = "https://github.com/sjtrny/forecast_dash/"
    github_patch_url = github_home_url + "commit/" + git_hash

    return dbc.Col(
        [
            dbc.Card(
                [
                    dbc.CardHeader(git_time),
                    dbc.CardBody(
                        [
                            html.H6(git_subject, className="card-title"),
                            html.P(f"by {git_author}", className="card-text"),
                        ]
                    ),
                    dbc.CardFooter(
                        [
                            dbc.CardLink(git_shorthash, href=github_patch_url),
                        ]
                    ),
                ],
                color="dark",
                outline=True,
            ),
            html.P(
                [
                    html.A("Development homepage", href=github_home_url),
                    " on github.",
                ]
            ),
        ]
    )


def footer():
    from app import nav_routes

    return [
        dbc.Row(dbc.Col(html.Hr(style={'margin-top': '64px'}), lg=12),),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Nav([
                            dbc.NavItem(dbc.NavLink(x[1], href=x[2], external_link=True))
                            for x in nav_routes
                        ], vertical="md",)
                    ],
                    lg=7,
                ),
                component_git_version(),
            ]
        , style={"margin-bottom": "64px"})
    ]


class BootstrapApp(dash.Dash, ABC):
    def __init__(self, name, server, url_base_pathname):

        external_stylesheets = [dbc.themes.BOOTSTRAP]

        super().__init__(
            name=name,
            server=server,
            url_base_pathname=url_base_pathname,
            external_stylesheets=external_stylesheets,
        )

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

    @property
    @classmethod
    @abstractmethod
    def title(cls):
        return NotImplementedError

    def setup(self):
        self.title = type(self).title

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
                    + footer(),
                    style={"margin-bottom": "64px"},
                ),
            ]
        )
