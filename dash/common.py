from abc import ABC, abstractmethod

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

import os

header = [
    dbc.NavbarSimple(
        children=[
            # dbc.DropdownMenu(
            #     children=[
            #         dbc.DropdownMenuItem(
            #             html.A(
            #                 "Australian Economic Indicators",
            #                 href="/filter?tags=Australia&tags=Economic",
            #             )
            #         ),
            #         dbc.DropdownMenuItem(
            #             html.A(
            #                 "US Economic Indicators",
            #                 href="/filter?tags=US&tags=Economic",
            #             )
            #         ),
            #         dbc.DropdownMenuItem(
            #             html.A(
            #                 "UK Economic Indicators",
            #                 href="/filter?tags=UK&tags=Economic",
            #             )
            #         ),
            #     ],
            #     nav=True,
            #     in_navbar=True,
            #     label="Economic Forecasts",
            #     disabled=False,
            # ),
            # dbc.DropdownMenu(
            #     children=[
            #         dbc.DropdownMenuItem(
            #             [
            #                 html.A(
            #                     "Australian Financial Indicators",
            #                     href="/filter?tags=Australia&tags=Financial",
            #                 )
            #             ]
            #         ),
            #         dbc.DropdownMenuItem(
            #             [
            #                 html.A(
            #                     "US Financial Indicators",
            #                     href="/filter?tags=US&tags=Financial",
            #                 )
            #             ]
            #         ),
            #         dbc.DropdownMenuItem(
            #             [
            #                 html.A(
            #                     "UK Financial Indicators",
            #                     href="/filter?tags=UK,Financial",
            #                 )
            #             ]
            #         ),
            #     ],
            #     nav=True,
            #     in_navbar=True,
            #     label="Financial Forecasts",
            # ),
            dbc.NavItem(
                dbc.NavLink(
                    "Find a Series", href="/search", external_link=True
                )
            ),
            dbc.NavItem(
                dbc.NavLink(
                    "Leaderboard", href="/leaderboard", external_link=True
                )
            ),
            dbc.NavItem(dbc.NavLink("Blog", href="/blog", external_link=True)),
            dbc.NavItem(
                dbc.NavLink(
                    "Methodology", href="/methodology", external_link=True
                )
            ),
            dbc.NavItem(
                dbc.NavLink("About", href="/about", external_link=True)
            ),
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

    return dbc.Col([
        html.Hr(),
        html.H3("Current Version"),
        dbc.Card([
            dbc.CardHeader(git_time),
            dbc.CardBody(
                [
                    html.H5(git_subject, className="card-title"),
                    html.P(f"by {git_author}", className="card-text"),
                ]
            ),
            dbc.CardFooter([
                dbc.CardLink(git_shorthash, href=github_patch_url),
            ]),
        ], color="dark", outline=True),
        html.P(
            [
                html.A(
                    "Development homepage", href=github_home_url
                ),
                " on github.",
            ]
        ),
    ])

footer = [
    dbc.Row([

        dbc.Col([
            html.Hr(),
            html.H3("Sections"),
            dbc.ListGroup([
                dbc.ListGroupItem(html.A("Find a Series", href="/search")),
                dbc.ListGroupItem(html.A("Leaderboard", href="/leaderboard")),
                dbc.ListGroupItem(html.A("Blog", href="/blog")),
                dbc.ListGroupItem(html.A("Methodology", href="/methodology")),
                dbc.ListGroupItem(html.A("About", href="/about")),
            ]),
        ], lg=7),
        component_git_version()
    ])
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
            header
            + [
                dcc.Location(id="url", refresh=False),
                dbc.Container(
                    [
                        breadcrumb_layout(
                            [("Home", "/"), (f"{self.title}", "")]
                        ),
                        dcc.Markdown(type(self).markdown),
                    ]
                ),
            ]
        )
