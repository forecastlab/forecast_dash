import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from urllib.parse import urlparse, parse_qsl, urlencode
from dash.exceptions import PreventUpdate
import dash
import pickle
import json
import dash_bootstrap_components as dbc
from abc import ABC, abstractmethod
import re
import ast

header = [
    dbc.NavbarSimple(
        children=[
            dbc.DropdownMenu(
                children=[
                    dbc.DropdownMenuItem(
                        html.A(
                            "Australian Economic Indicators",
                            href="/filter?tags=Australia,Economic",
                        )
                    ),
                    dbc.DropdownMenuItem(
                        html.A(
                            "US Economic Indicators",
                            href="/filter?tags=USA,Economic",
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
                                href="/filter?tags=Australia,Financial",
                            )
                        ]
                    ),
                    dbc.DropdownMenuItem(
                        [
                            html.A(
                                "US Financial Indicators",
                                href="/filter?tags=USA,Financial",
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


def parse_state(url):
    parse_result = urlparse(url)
    params = parse_qsl(parse_result.query)
    state = dict(params)
    return state


def get_forecast_plot_data(series_df, forecast_df):

    line_history = go.Scatter(
        x=series_df["date"],
        y=series_df["value"],
        name="Historical",
        mode="lines+markers",
    )

    forecast_error_x = list(forecast_df.index) + list(
        reversed(forecast_df.index)
    )
    forecast_error_x = [x.to_pydatetime() for x in forecast_error_x]

    error_50 = go.Scatter(
        x=forecast_error_x,
        y=list(forecast_df["UB_50"]) + list(reversed(forecast_df["LB_50"])),
        fill="tozeroy",
        fillcolor="rgba(0,100,80,0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        name="50% CI",
    )

    error_75 = go.Scatter(
        x=forecast_error_x,
        y=list(forecast_df["UB_75"]) + list(reversed(forecast_df["LB_75"])),
        fill="tozeroy",
        fillcolor="rgba(0,176,246,0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        name="75% CI",
    )

    error_95 = go.Scatter(
        x=forecast_error_x,
        y=list(forecast_df["UB_95"]) + list(reversed(forecast_df["LB_95"])),
        fill="tozeroy",
        fillcolor="rgba(231,107,243,0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% CI",
    )

    line_forecast = go.Scatter(
        x=forecast_df.index,
        y=forecast_df["FORECAST"],
        name="Forecast",
        mode="lines+markers",
        line=dict(dash="2px"),
    )

    data = [line_history, error_95, error_75, error_50, line_forecast]

    return data


def get_thumbnail_figure(data_dict):

    series_df = data_dict["series_df"].iloc[-16:, :]
    forecast_df = data_dict["forecast_df"]

    data = get_forecast_plot_data(series_df, forecast_df)

    layout = go.Layout(
        title=data_dict["data_source_dict"]["title"] + " Forecast",
        height=480,
        showlegend=False,
    )

    return go.Figure(data, layout)


def get_series_figure(data_dict):

    series_df = data_dict["series_df"]
    forecast_df = data_dict["forecast_df"]

    data = get_forecast_plot_data(series_df, forecast_df)

    time_difference_forecast_to_start = (
        forecast_df.index[-1].to_pydatetime()
        - series_df["date"][0].to_pydatetime()
    )

    layout = go.Layout(
        title=data_dict["data_source_dict"]["title"] + " Forecast",
        height=720,
        xaxis=dict(
            fixedrange=True,
            type="date",
            range=[
                series_df["date"]
                .iloc[-16]
                .to_pydatetime(),  # Recent point in history
                forecast_df.index[-1].to_pydatetime(),  # End of forecast range
            ],
            rangeselector=dict(
                buttons=list(
                    [
                        dict(
                            count=5,
                            label="5y",
                            step="year",
                            stepmode="backward",
                        ),
                        dict(
                            count=10,
                            label="10y",
                            step="year",
                            stepmode="backward",
                        ),
                        dict(
                            count=time_difference_forecast_to_start.days,
                            label="all",
                            step="day",
                            stepmode="backward",
                        ),
                    ]
                )
            ),
            rangeslider=dict(
                visible=True,
                range=[
                    series_df["date"][0].to_pydatetime(),
                    forecast_df.index[-1].to_pydatetime(),
                ],
            ),
        ),
        yaxis=dict(
            # Will disable all zooming and movement controls if True
            fixedrange=True,
            autorange=True,
        ),
    )

    return go.Figure(data, layout)


def get_series_data(title):
    f = open(f"forecasts/{title}.pkl", "rb")
    data_dict = pickle.load(f)
    return data_dict


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


class Index(BootstrapApp):
    def setup(self):

        self.title = "Business Forecast Lab"

        showcase_item_titles = [
            "Australian GDP",
            "Australian Underemployment",
            "Australian Inflation (CPI)",
            "Australian Unemployment",
        ]

        def layout_func():

            showcase_list = []

            for item_title in showcase_item_titles:

                series_data = get_series_data(item_title)
                thumbnail_figure = get_thumbnail_figure(series_data)
                showcase_list.append(
                    dbc.Col(
                        [
                            html.A(
                                [
                                    dcc.Graph(
                                        id=item_title,
                                        figure=thumbnail_figure,
                                        config={
                                            "displayModeBar": False,
                                            "staticPlot": False,
                                        },
                                    )
                                ],
                                href=f"/series?title={item_title}",
                            )
                        ],
                        lg=6,
                        sm=12,
                    )
                )

            showcase_div = dbc.Row(showcase_list, className="row")

            return html.Div(
                header
                + [
                    dcc.Location(id="url", refresh=False),
                    dbc.Container(
                        [
                            html.H2(
                                "Featured",
                                style={"text-align": "center"},
                                className="mt-3",
                            ),
                            showcase_div,
                        ]
                    ),
                ]
            )

        self.layout = layout_func


class Series(BootstrapApp):
    def _serve_series(self, title):

        series_data = get_series_data(title)
        series_figure = get_series_figure(series_data)

        series_graph = dcc.Graph(
            figure=series_figure,
            config={
                "modeBarButtonsToRemove": [
                    "sendDataToCloud",
                    "autoScale2d",
                    "hoverClosestCartesian",
                    "hoverCompareCartesian",
                    "lasso2d",
                    "select2d",
                    "toggleSpikelines",
                ],
                "displaylogo": False,
            },
        )

        dataframe = series_data["forecast_df"]

        table = dbc.Table(
            # Header
            [
                html.Thead(
                    html.Tr(
                        [html.Th("Date")]
                        + [html.Th(col) for col in dataframe.columns]
                    )
                )
            ]
            +
            # Body
            [
                html.Tbody(
                    [
                        html.Tr(
                            [html.Td(dataframe.index[i])]
                            + [
                                html.Td(dataframe.iloc[i][col].round(4))
                                for col in dataframe.columns
                            ]
                        )
                        for i in range(1, len(dataframe))
                    ]
                )
            ]
        )

        return [
            dbc.Nav(
                [
                    html.Ol(
                        [
                            html.Li(
                                html.A("Home", href="/"),
                                className="breadcrumb-item",
                            ),
                            html.Li(
                                f"Series: {title}",
                                className="breadcrumb-item active",
                            ),
                        ],
                        className="breadcrumb",
                    )
                ],
                navbar=True,
            ),
            html.H3(),
            series_graph,
            table,
        ]

    def setup(self):

        self.title = "Series"

        self.layout = html.Div(
            header
            + [
                dcc.Location(id="url", refresh=False),
                dbc.Container(
                    [
                        dbc.Row(
                            [dbc.Col([html.Div(id="dynamic_content")], lg=12)]
                        )
                    ]
                ),
            ]
        )

        @self.callback(
            Output("dynamic_content", "children"), [Input("url", "href")]
        )
        def display_value(value):
            # Put this in to avoid an Exception due to weird Location component
            # behaviour
            if value is None:
                raise PreventUpdate

            parse_result = parse_state(value)

            if "title" in parse_result:
                return self._serve_series(parse_result["title"])
            else:
                raise PreventUpdate


def apply_default_value(params):
    def wrapper(func):
        def apply_value(*args, **kwargs):
            if "id" in kwargs and kwargs["id"] in params:

                # For inputs that accept value vs values
                if "value" in kwargs:
                    key = "value"
                    try:
                        kwargs[key] = ast.literal_eval(params[kwargs["id"]])
                    except Exception:
                        kwargs[key] = params[kwargs["id"]]
                else:
                    key = "values"
                    # It will either be a comma seperated string or a list-like
                    try:
                        kwargs[key] = ast.literal_eval(params[kwargs["id"]])
                    except Exception:
                        kwargs[key] = params[kwargs["id"]].split(",")

            return func(*args, **kwargs)

        return apply_value

    return wrapper


class Filter(BootstrapApp):
    def match_names(self, name_input, series_dicts):
        # This doesn't need to be an instance method

        matched_series_names = []

        name_terms = "|".join(name_input.split(" "))

        for data_source_dict in series_dicts.values():

            re_results = re.search(
                name_terms, data_source_dict["title"], re.IGNORECASE
            )
            if re_results is not None:
                matched_series_names.append(data_source_dict["title"])

        return matched_series_names

    def match_tags(self, tags, series_dicts):
        # This doesn't need to be an instance method

        matched_series_names = []

        if type(tags) == str:
            tags = tags.split(",")

        tags = set(tags)

        for data_source_dict in series_dicts.values():

            if tags.issubset(set(data_source_dict["tags"])):
                matched_series_names.append(data_source_dict["title"])

        return matched_series_names

    def setup(self):

        self.config.suppress_callback_exceptions = True

        self.title = "Filter"

        self.layout = html.Div(
            header
            + [
                dcc.Location(id="url", refresh=False),
                dbc.Container(
                    [
                        html.H3("Filter"),
                        dbc.Row(
                            [
                                dbc.Col(id="filter_panel", lg=3, sm=3),
                                dbc.Col(
                                    [
                                        html.H4("Results"),
                                        dcc.Loading(
                                            html.Div(id="filter_results")
                                        ),
                                    ],
                                    lg=9,
                                    sm=9,
                                ),
                            ]
                        ),
                    ]
                ),
            ]
        )

        def filter_panel_children(params):

            children = [
                dbc.FormGroup(
                    [
                        dbc.Label("Name"),
                        apply_default_value(params)(dbc.Input)(
                            id="name",
                            placeholder="Name of a series...",
                            type="text",
                            value="",
                        ),
                        dbc.FormText("Type something in the box above"),
                    ]
                ),
                dbc.FormGroup(
                    [
                        dbc.Label("Tags"),
                        apply_default_value(params)(dbc.Checklist)(
                            options=[
                                {"label": "Australia", "value": "Australia"},
                                {"label": "USA", "value": "USA"},
                                {"label": "Economic", "value": "Economic"},
                                {"label": "Financial", "value": "Financial"},
                            ],
                            values=[],
                            id="tags",
                        ),
                    ]
                ),
            ]

            return children

        @self.callback(
            Output("filter_panel", "children"), [Input("url", "href")]
        )
        def display_value(value):
            # Put this in to avoid an Exception due to weird Location component
            # behaviour
            if value is None:
                raise PreventUpdate

            parse_result = parse_state(value)

            return filter_panel_children(parse_result)

        value_component_ids = ["name"]

        values_component_ids = ["tags"]

        @self.callback(
            Output("url", "search"),
            inputs=[Input(i, "value") for i in value_component_ids]
            + [Input(i, "values") for i in values_component_ids],
        )
        def update_url_state(*values):

            state = urlencode(
                dict(zip(value_component_ids + values_component_ids, values))
            )

            return f"?{state}"

        @self.callback(
            Output("filter_results", "children"),
            [Input(i, "value") for i in value_component_ids]
            + [Input(i, "values") for i in values_component_ids],
        )
        def filter_results(name, tags):

            # Searching by AND-ing conditions together

            # Search
            # - names
            # - tags

            data_sources_json_file = open("data_sources.json")
            series_list = json.load(data_sources_json_file)
            data_sources_json_file.close()

            series_dicts = {}

            for series_dict in series_list:
                series_dicts[series_dict["title"]] = series_dict

            list_filter_matches = []
            if name == "":
                list_filter_matches.append(set(series_dicts.keys()))
            else:
                matched_series_names = self.match_names(name, series_dicts)
                list_filter_matches.append(set(matched_series_names))
            if tags == "":
                list_filter_matches.append(set(series_dicts.keys()))
            else:
                matched_series_names = self.match_tags(tags, series_dicts)
                list_filter_matches.append(set(matched_series_names))

            unique_series_titles = list(
                sorted(set.intersection(*list_filter_matches))
            )

            results_list = []

            for item_title in unique_series_titles:
                series_data = get_series_data(item_title)
                thumbnail_figure = get_thumbnail_figure(series_data)

                results_list.append(
                    html.Div(
                        [
                            html.A(
                                [
                                    html.H5(item_title),
                                    dcc.Graph(
                                        id=item_title,
                                        figure=thumbnail_figure,
                                        config={
                                            "displayModeBar": False,
                                            "staticPlot": True,
                                        },
                                        className="six columns",
                                    ),
                                ],
                                href=f"/series?title={item_title}",
                            ),
                            html.Hr(),
                        ]
                    )
                )

            if len(unique_series_titles) > 0:
                results = [
                    html.P(
                        f"{len(unique_series_titles)} result{'s' if len(unique_series_titles) > 1 else ''} found"
                    ),
                    html.Div(results_list),
                ]
            else:
                results = [html.P("No results found")]

            return results


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
                dbc.Container([dcc.Markdown(type(self).markdown)]),
            ]
        )


class Methodology(MarkdownApp):

    title = "Methodology"

    markdown = """
# Methodology

**This page is under construction.**

It will contain the description of the models and other aspects of the
methodology used to forecast the time series.

While we are busy with this document, we recommend “Forecasting: Principles
and Practice” textbook freely available at
[otexts.com/fpp2/](https://otexts.com/fpp2/)
    """


class About(BootstrapApp):

    title = "About"

    def setup(self):

        self.layout = html.Div(
            header
            + [
                dcc.Location(id="url", refresh=False),
                dbc.Container(
                    [
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
                                    [
                                        html.H2("Members"),
                                        html.P(
                                            "The Business Forecast Lab was established in ...."
                                        ),
                                        html.H4(
                                            "Andrey Vasnev", className="mt-3"
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Img(
                                                            src="https://business.sydney.edu.au/__data/assets/image/0006/170556/vasnev.png",
                                                            height="200px",
                                                        )
                                                    ],
                                                    lg=2,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.P(
                                                            "Andrey Vasnev (Perm, 1976) graduated in Applied Mathematics from Moscow State University in 1998. In 2001 he completed his Master's degree in Economics in the New Economic School, Moscow. In 2006 he received Ph.D. degree in Economics from the Department of Econometrics and Operations Research at Tilburg University under the supervision of Jan R. Magnus. He worked as a credit risk analyst in ABN AMRO bank before joining the University of Sydney."
                                                        ),
                                                        html.A(
                                                            "https://business.sydney.edu.au/staff/andrey.vasnev",
                                                            href="https://business.sydney.edu.au/staff/andrey.vasnev",
                                                        ),
                                                    ],
                                                    lg=9,
                                                ),
                                            ]
                                        ),
                                        html.H4(
                                            "Richard Gerlach", className="mt-3"
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Img(
                                                            src="https://business.sydney.edu.au/__data/assets/image/0003/170553/RichardGerlach.jpg",
                                                            height="200px",
                                                        )
                                                    ],
                                                    lg=2,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.P(
                                                            "Richard Gerlach's research interests lie mainly in financial econometrics and time series. His work has concerned developing time series models for measuring, forecasting and managing risk in financial markets as well as computationally intensive Bayesian methods for inference, diagnosis, forecasting and model comparison for these models. Recent focus has been on nonlinear threshold heteroskedastic models for volatility, Value-at-Risk and Expected Shortfall forecasting. He has developed structural break and intervention detection tools for use in state space models; also has an interest in estimating logit models incorporating misclassification and variable selection. His applied work has involved forecasting risk levels during and after the Global Financial Crisis; assessing asymmetry in major international stock markets, in response to local and exogenous factors; co-integration analysis assessing the effect of the Asian financial crisis on long term relationships between international real estate investment markets; stock selection for financial investment using logit models; option pricing and hedging involving barriers; and factors influencing the 2004 Federal election."
                                                        ),
                                                        html.P(
                                                            "His research papers have been published in Journal of the American Statistical Association, Journal of Business and Economic Statistics, Journal of Time Series Analysis and the International Journal of Forecasting. He has been an invited speaker and regular presenter at international conferences such as the International conference for Computational and Financial Econometrics, the International Symposium on Forecasting and the International Statistical Institute sessions."
                                                        ),
                                                        html.A(
                                                            "https://business.sydney.edu.au/staff/richard.gerlach",
                                                            href="https://business.sydney.edu.au/staff/richard.gerlach",
                                                        ),
                                                    ],
                                                    lg=9,
                                                ),
                                            ]
                                        ),
                                        html.H4("Chao Wang", className="mt-3"),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Img(
                                                            src="https://business.sydney.edu.au/__data/assets/image/0012/279678/wang.jpg",
                                                            height="200px",
                                                        )
                                                    ],
                                                    lg=2,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.P(
                                                            "Dr Chao Wang received his PhD degree in Econometrics from The University of Sydney. He has two master degrees major in Machine Learning & Data Mining from Helsinki University of Technology and Mechatronic Engineering from Beijing Institute of Technology respectively."
                                                        ),
                                                        html.P(
                                                            "Chao Wang’s main research interests are financial econometrics and time series modelling. He has developed a series of parametric and non-parametric volatility models incorporating intra-day and high frequency volatility measures (realized variance, realized range, etc) applied on the financial market risk forecasting, employing Bayesian adaptive Markov chain Monte Carlo estimation. His work has also considered different techniques, including scaling and sub-sampling, to deal with the micro-structure noisy of the high frequency volatility measures. Further, Chao’s research interests also include big data, machine learning and data mining, text mining, etc."
                                                        ),
                                                        html.A(
                                                            "https://business.sydney.edu.au/staff/chao.wang",
                                                            href="https://business.sydney.edu.au/staff/chao.wang",
                                                        ),
                                                    ],
                                                    lg=9,
                                                ),
                                            ]
                                        ),
                                    ],
                                    lg=12,
                                )
                            ]
                        ),
                    ],
                    className="mb-5",
                ),
            ]
        )
