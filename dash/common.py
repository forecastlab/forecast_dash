from abc import ABC, abstractmethod

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

header = [
    dbc.NavbarSimple(
        children=[
            dbc.DropdownMenu(
                children=[
                    dbc.DropdownMenuItem(
                        html.A(
                            "Australian Economic Indicators",
                            href="/filter?tags=Australia&tags=Economic",
                        )
                    ),
                    dbc.DropdownMenuItem(
                        html.A(
                            "US Economic Indicators",
                            href="/filter?tags=US&tags=Economic",
                        )
                    ),
                    dbc.DropdownMenuItem(
                        html.A(
                            "UK Economic Indicators",
                            href="/filter?tags=UK&tags=Economic",
                        )
                    ),
                ],
                nav=True,
                in_navbar=True,
                label="Economic Forecasts",
                disabled=False,
            ),
            dbc.DropdownMenu(
                children=[
                    dbc.DropdownMenuItem(
                        [
                            html.A(
                                "Australian Financial Indicators",
                                href="/filter?tags=Australia&tags=Financial",
                            )
                        ]
                    ),
                    dbc.DropdownMenuItem(
                        [
                            html.A(
                                "US Financial Indicators",
                                href="/filter?tags=US&tags=Financial",
                            )
                        ]
                    ),
                    dbc.DropdownMenuItem(
                        [
                            html.A(
                                "UK Financial Indicators",
                                href="/filter?tags=UK,Financial",
                            )
                        ]
                    ),
                ],
                nav=True,
                in_navbar=True,
                label="Financial Forecasts",
            ),
            dbc.NavItem(
                dbc.NavLink("Filter", href="/filter", external_link=True)
            ),
            dbc.NavItem(
                dbc.NavLink("Stats", href="/stats", external_link=True)
            ),
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
