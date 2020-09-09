import ast
import json
import pickle
import re
from functools import wraps
from urllib.parse import urlparse, parse_qs, urlencode

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from common import BootstrapApp, header, breadcrumb_layout
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate


def dash_kwarg(inputs):
    def accept_func(func):
        @wraps(func)
        def wrapper(*args):
            input_names = [item.component_id for item in inputs]
            kwargs_dict = dict(zip(input_names, args))
            return func(**kwargs_dict)

        return wrapper

    return accept_func


def parse_state(url):
    parse_result = urlparse(url)
    return parse_qs(parse_result.query)


def get_forecast_plot_data(series_df, forecast_df):

    line_history = go.Scatter(
        x=series_df.index,
        y=series_df["value"],
        name="Historical",
        mode="lines+markers",
        line=dict(color="rgb(0, 0, 0)"),
    )

    forecast_error_x = list(forecast_df.index) + list(
        reversed(forecast_df.index)
    )
    forecast_error_x = [x.to_pydatetime() for x in forecast_error_x]

    error_50 = go.Scatter(
        x=forecast_error_x,
        y=list(forecast_df["UB_50"]) + list(reversed(forecast_df["LB_50"])),
        fill="tozeroy",
        fillcolor="rgb(226, 87, 78)",
        line=dict(color="rgba(255,255,255,0)"),
        name="50% CI",
    )

    error_75 = go.Scatter(
        x=forecast_error_x,
        y=list(forecast_df["UB_75"]) + list(reversed(forecast_df["LB_75"])),
        fill="tozeroy",
        fillcolor="rgb(234, 130, 112)",
        line=dict(color="rgba(255,255,255,0)"),
        name="75% CI",
    )

    error_95 = go.Scatter(
        x=forecast_error_x,
        y=list(forecast_df["UB_95"]) + list(reversed(forecast_df["LB_95"])),
        fill="tozeroy",
        fillcolor="rgb(243, 179, 160)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% CI",
    )

    line_forecast = go.Scatter(
        x=forecast_df.index,
        y=forecast_df["forecast"],
        name="Forecast",
        mode="lines",
        line=dict(color="rgb(0,0,0)", dash="2px"),
    )

    data = [error_95, error_75, error_50, line_forecast, line_history]

    return data


def get_forecast_shapes(forecast_df):

    shapes = [
        {
            "type": "rect",
            # x-reference is assigned to the x-values
            "xref": "x",
            # y-reference is assigned to the plot paper [0,1]
            "yref": "paper",
            "x0": forecast_df.index[0],
            "y0": 0,
            "x1": forecast_df.index[-1],
            "y1": 1,
            "fillcolor": "rgba(0,0,0)",
            "opacity": 0.1,
            "line": {"width": 0},
            "layer": "below",
        }
    ]

    return shapes


def select_best_model(data_dict):

    # Extract ( model_name, cv_score ) for each model.
    all_models = []
    all_cv_scores = []
    for model_name, forecast_df in data_dict["all_forecasts"].items():
        all_models.append(model_name)
        all_cv_scores.append(forecast_df["cv_score"])

    # Select the best model.
    model_name = all_models[np.argmin(all_cv_scores)]

    return model_name


def get_thumbnail_figure(data_dict):

    model_name = select_best_model(data_dict)
    series_df = data_dict["downloaded_dict"]["series_df"].iloc[-16:, :]
    forecast_df = data_dict["all_forecasts"][model_name]["forecast_df"]

    data = get_forecast_plot_data(series_df, forecast_df)
    shapes = get_forecast_shapes(forecast_df)

    title = data_dict["data_source_dict"]["title"]
    layout = go.Layout(
        title={"text": title, "xanchor": "auto"},
        height=480,
        showlegend=False,
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True),
        shapes=shapes,
    )

    return go.Figure(data, layout)


def get_series_figure(data_dict, model_name):

    series_df = data_dict["downloaded_dict"]["series_df"]
    forecast_df = data_dict["all_forecasts"][model_name]["forecast_df"]

    data = get_forecast_plot_data(series_df, forecast_df)
    shapes = get_forecast_shapes(forecast_df)

    time_difference_forecast_to_start = (
        forecast_df.index[-1].to_pydatetime()
        - series_df.index[0].to_pydatetime()
    )

    title = data_dict["data_source_dict"]["title"]
    layout = go.Layout(
        title=title,
        height=720,
        xaxis=dict(
            fixedrange=True,
            type="date",
            range=[
                series_df.index[
                    -16
                ].to_pydatetime(),  # Recent point in history
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
                    series_df.index[0].to_pydatetime(),
                    forecast_df.index[-1].to_pydatetime(),
                ],
            ),
        ),
        yaxis=dict(
            # Will disable all zooming and movement controls if True
            fixedrange=True,
            autorange=True,
        ),
        shapes=shapes,
    )

    return go.Figure(data, layout)


def get_forecast_data(title):
    f = open(f"../data/forecasts/{title}.pkl", "rb")
    data_dict = pickle.load(f)
    return data_dict


class Index(BootstrapApp):
    def setup(self):

        self.title = "Business Forecast Lab"

        showcase_item_titles = [
            "Australian GDP Growth",
            "Australian Inflation (CPI)",
            "Australian Unemployment",
            "Australian Underemployment",
            "US GDP Growth",
            "US Unemployment",
            "UK Inflation (RPI)",
            "UK Unemployment",
        ]

        def layout_func():

            showcase_list = []

            for item_title in showcase_item_titles:

                # If data not present for any reason skip this
                try:
                    series_data = get_forecast_data(item_title)
                except FileNotFoundError:
                    continue

                url_title = urlencode({"title": item_title})
                thumbnail_figure = get_thumbnail_figure(series_data)
                showcase_list.append(
                    dbc.Col(
                        [
                            html.A(
                                [
                                    dcc.Graph(
                                        id=item_title,
                                        figure=thumbnail_figure,
                                        config={"displayModeBar": False},
                                    )
                                ],
                                href=f"/series?{url_title}",
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
                    dbc.Jumbotron(
                        dbc.Container(
                            [
                                html.H1("Our Mission", className="display-4"),
                                # html.Hr(),
                                html.P(
                                    "To make forecasting accessible to everyone by providing:",
                                    className="lead",
                                ),
                                html.Ol(
                                    [
                                        html.Li(
                                            "up to date forecasts for common or important time series.",
                                            className="lead",
                                        ),
                                        html.Li(
                                            "evaluations and comparisons of forecasting methods.",
                                            className="lead",
                                        ),
                                    ]
                                ),
                            ]
                        ),
                        fluid=True,
                    ),
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


def location_ignore_null(inputs, location_id):
    def accept_func(func):
        @wraps(func)
        def wrapper(*args):
            input_names = [item.component_id for item in inputs]
            kwargs_dict = dict(zip(input_names, args))

            if kwargs_dict[location_id] is None:
                raise PreventUpdate

            return func(*args)

        return wrapper

    return accept_func


class Series(BootstrapApp):
    def setup(self):

        self.title = "Series"

        self.layout = html.Div(
            header
            + [
                dcc.Location(id="url", refresh=False),
                dbc.Container(
                    [
                        breadcrumb_layout([("Home", "/"), ("Series", "")]),
                        dcc.Loading(
                            dbc.Row([dbc.Col(id="series_graph", lg=12)])
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.FormGroup(
                                            [
                                                dbc.Label("Forecast Method"),
                                                dcc.Dropdown(
                                                    id="model_selector",
                                                    clearable=False,
                                                ),
                                            ]
                                        ),
                                        dcc.Loading(
                                            html.Div(id="meta_data_list")
                                        ),
                                    ],
                                    lg=6,
                                ),
                                dbc.Col(
                                    [
                                        dbc.FormGroup(
                                            [
                                                dcc.Dropdown(
                                                    options=[
                                                        {
                                                            "label": "Forecast",
                                                            "value": "Forecast",
                                                        },
                                                        {
                                                            "label": "50% CI",
                                                            "value": "CI_50",
                                                        },
                                                        {
                                                            "label": "75% CI",
                                                            "value": "CI_75",
                                                        },
                                                        {
                                                            "label": "95% CI",
                                                            "value": "CI_95",
                                                        },
                                                    ],
                                                    value="Forecast",
                                                    clearable=False,
                                                    id="forecast_table_selector",
                                                ),
                                            ]
                                        ),
                                        dcc.Loading(
                                            html.Div(id="forecast_table")
                                        ),
                                    ],
                                    lg=6,
                                ),
                            ]
                        ),
                    ]
                ),
            ]
        )

        def series_input(inputs, location_id="url"):
            def accept_func(func):
                @wraps(func)
                def wrapper(*args):
                    input_names = [item.component_id for item in inputs]
                    kwargs_dict = dict(zip(input_names, args))

                    parse_result = parse_state(kwargs_dict[location_id])

                    if "title" in parse_result:
                        title = parse_result["title"][0]
                        series_data_dict = get_forecast_data(title)

                        del kwargs_dict[location_id]
                        return func(series_data_dict, **kwargs_dict)
                    else:
                        raise PreventUpdate

                return wrapper

            return accept_func

        inputs = [Input("url", "href")]

        @self.callback(Output("breadcrumb", "children"), inputs)
        @location_ignore_null(inputs, location_id="url")
        @series_input(inputs, location_id="url")
        def update_breadcrumb(series_data_dict):

            return series_data_dict["data_source_dict"]["title"]

        @self.callback(
            Output("series_graph", "children"),
            inputs + [Input("model_selector", "value")],
        )
        @location_ignore_null(inputs, location_id="url")
        @series_input(
            inputs + [Input("model_selector", "value")], location_id="url"
        )
        def update_series_graph(series_data_dict, **kwargs):

            model_name = kwargs["model_selector"]

            series_figure = get_series_figure(series_data_dict, model_name)

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

            return series_graph

        @self.callback(
            [
                Output("model_selector", "options"),
                Output("model_selector", "value"),
            ],
            inputs,
        )
        @location_ignore_null(inputs, location_id="url")
        @series_input(inputs, location_id="url")
        def update_model_selector(series_data_dict):

            best_model_name = select_best_model(series_data_dict)

            stats = get_forecast_data("statistics")
            all_methods = sorted(stats["models_used"])

            all_methods_dict = dict(zip(all_methods, all_methods))

            all_methods_dict[
                best_model_name
            ] = f"{best_model_name} - Best Model"

            model_select_options = [
                {"label": v, "value": k} for k, v in all_methods_dict.items()
            ]

            return model_select_options, best_model_name

        @self.callback(
            Output("meta_data_list", "children"),
            inputs + [Input("model_selector", "value")],
        )
        @location_ignore_null(inputs, location_id="url")
        @series_input(
            inputs + [Input("model_selector", "value")], location_id="url"
        )
        def update_meta_data_list(series_data_dict, **kwargs):

            model_name = kwargs["model_selector"]

            model_description = series_data_dict["all_forecasts"][model_name][
                "model_description"
            ]
            if model_description == model_name:
                model_description = ""

            model_cv_score = series_data_dict["all_forecasts"][model_name][
                "cv_score"
            ]

            return dbc.ListGroup(
                [
                    dbc.ListGroupItem(
                        [
                            dbc.ListGroupItemHeading("Model Details"),
                            dbc.ListGroupItemText(
                                [
                                    html.P(model_name),
                                    html.P(model_description),
                                    html.P("CV score: %f" % model_cv_score),
                                ]
                            ),
                        ]
                    ),
                    dbc.ListGroupItem(
                        [
                            dbc.ListGroupItemHeading("Forecast Updated At"),
                            dbc.ListGroupItemText(
                                series_data_dict["forecasted_at"].strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                )
                            ),
                        ]
                    ),
                    dbc.ListGroupItem(
                        [
                            dbc.ListGroupItemHeading("Data Collected At"),
                            dbc.ListGroupItemText(
                                series_data_dict["downloaded_dict"][
                                    "downloaded_at"
                                ].strftime("%Y-%m-%d %H:%M:%S")
                            ),
                        ]
                    ),
                    dbc.ListGroupItem(
                        [
                            dbc.ListGroupItemHeading("Data Source"),
                            dbc.ListGroupItemText(
                                [
                                    html.A(
                                        series_data_dict["data_source_dict"][
                                            "url"
                                        ],
                                        href=series_data_dict[
                                            "data_source_dict"
                                        ]["url"],
                                    )
                                ]
                            ),
                        ]
                    ),
                ]
            )

        @self.callback(
            Output("forecast_table", "children"),
            inputs
            + [
                Input("forecast_table_selector", "value"),
                Input("model_selector", "value"),
            ],
        )
        @location_ignore_null(inputs, location_id="url")
        @series_input(
            inputs
            + [
                Input("forecast_table_selector", "value"),
                Input("model_selector", "value"),
            ],
            location_id="url",
        )
        def update_forecast_table(series_data_dict, **kwargs):

            selected_column_map = {
                "Forecast": ["Forecast"],
                "CI_50": ["LB_50", "UB_50"],
                "CI_75": ["LB_75", "UB_75"],
                "CI_95": ["LB_95", "UB_95"],
            }

            model_name = kwargs["model_selector"]

            dataframe = series_data_dict["all_forecasts"][model_name][
                "forecast_df"
            ]

            column_name_map = {"forecast": "Forecast"}

            dataframe = dataframe.rename(column_name_map, axis=1)[
                selected_column_map[kwargs["forecast_table_selector"]]
            ].round(4)

            dataframe.index = dataframe.index.strftime("%Y-%m-%d %H:%M:%S")

            table = dbc.Table.from_dataframe(
                dataframe, index=True, index_label="Date"
            )

            return table


def apply_default_value(params):
    def wrapper(func):
        def apply_value(*args, **kwargs):
            if "id" in kwargs and kwargs["id"] in params:
                key = "value"
                try:
                    kwargs[key] = ast.literal_eval(params[kwargs["id"]])
                except Exception:
                    kwargs[key] = params[kwargs["id"]]

            return func(*args, **kwargs)

        return apply_value

    return wrapper


class Stats(BootstrapApp):
    def setup(self):

        self.title = "Statistics"

        def layout_func():

            try:
                stats = get_forecast_data("statistics")
                all_methods = stats["models_used"]
            except FileNotFoundError:
                all_methods = []

            data_sources_json_file = open("../shared_config/data_sources.json")
            source_series_list = json.load(data_sources_json_file)
            data_sources_json_file.close()

            forecast_series_dicts = {}

            for series_dict in source_series_list:
                try:
                    forecast_series_dicts[
                        series_dict["title"]
                    ] = get_forecast_data(series_dict["title"])
                except FileNotFoundError:
                    continue

            chosen_methods = []
            for series_title, forecast_data in forecast_series_dicts.items():
                model_name = select_best_model(forecast_data)
                chosen_methods.append(model_name)

            stats_raw = pd.DataFrame({"Method": chosen_methods})

            unchosen_methods = list(set(all_methods) - set(chosen_methods))
            unchosen_counts = pd.Series(
                data=np.zeros(len(unchosen_methods)),
                index=unchosen_methods,
                name="Total",
            )

            counts = pd.DataFrame(
                stats_raw["Method"]
                .value_counts()
                .rename("Total")
                .append(unchosen_counts)
            )
            counts["Proportion"] = counts["Total"] / counts["Total"].sum()

            table = dbc.Table.from_dataframe(
                counts, index=True, index_label="Method"
            )

            # Apply URLS to index
            for row in table.children[1].children:
                state = urlencode(
                    {"methods": [row.children[0].children]}, doseq=True
                )
                row.children[0].children = html.A(
                    row.children[0].children, href=f"/filter/?{state}"
                )

            return html.Div(
                header
                + [
                    dcc.Location(id="url", refresh=False),
                    dbc.Container(
                        [
                            breadcrumb_layout(
                                [("Home", "/"), (f"{self.title}", "")]
                            ),
                            html.H2("Statistics"),
                            table,
                        ]
                    ),
                ]
            )

        self.layout = layout_func


def match_names(forecast_dicts, name_input):
    if not name_input or name_input == "":
        return set(forecast_dicts.keys())

    matched_series_names = []

    name_terms = "|".join(name_input.split(" "))

    for series_title, forecast_dict in forecast_dicts.items():

        re_results = re.search(name_terms, series_title, re.IGNORECASE)
        if re_results is not None:
            matched_series_names.append(series_title)

    return set(matched_series_names)


def match_tags(forecast_dicts, tags):
    if not tags or tags == "":
        return set(forecast_dicts.keys())

    matched_series_names = []

    if type(tags) == str:
        tags = tags.split(",")

    tags = set(tags)

    for series_title, forecast_dict in forecast_dicts.items():
        series_tags = forecast_dict["data_source_dict"]["tags"]

        if tags.issubset(set(series_tags)):
            matched_series_names.append(series_title)

    return set(matched_series_names)


def match_methods(forecast_dicts, methods):
    if not methods or methods == "":
        return set(forecast_dicts.keys())

    matched_series_names = []

    if type(methods) == str:
        methods = methods.split(",")

    methods = set(methods)

    for series_title, forecast_dict in forecast_dicts.items():

        if select_best_model(forecast_dict) in methods:
            matched_series_names.append(series_title)

    return set(matched_series_names)


class Filter(BootstrapApp):
    def setup(self):

        self.config.suppress_callback_exceptions = True

        self.title = "Filter"

        self.layout = html.Div(
            header
            + [
                dcc.Location(id="url", refresh=False),
                dbc.Container(
                    [
                        breadcrumb_layout([("Home", "/"), ("Filter", "")]),
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

        def filter_panel_children(params, tags, methods):

            children = [
                html.H4("Filters"),
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
                            options=[{"label": t, "value": t} for t in tags],
                            value=[],
                            id="tags",
                        ),
                    ]
                ),
                dbc.FormGroup(
                    [
                        dbc.Label("Method"),
                        apply_default_value(params)(dbc.Checklist)(
                            options=[
                                {"label": m, "value": m} for m in methods
                            ],
                            value=[],
                            id="methods",
                        ),
                    ]
                ),
            ]

            return children

        component_ids = ["name", "tags", "methods"]

        @self.callback(
            Output("filter_panel", "children"), [Input("url", "href")]
        )
        @location_ignore_null([Input("url", "href")], "url")
        def display_value(value):
            parse_result = parse_state(value)

            # Dynamically load tags
            data_sources_json_file = open("../shared_config/data_sources.json")
            series_list = json.load(data_sources_json_file)
            data_sources_json_file.close()

            all_tags = []

            for series_dict in series_list:
                all_tags.extend(series_dict["tags"])

            all_tags = sorted(set(all_tags))

            # Dynamically load methods
            stats = get_forecast_data("statistics")
            all_methods = stats["models_used"]

            return filter_panel_children(parse_result, all_tags, all_methods)

        @self.callback(
            Output("url", "search"),
            inputs=[Input(i, "value") for i in component_ids],
        )
        @dash_kwarg([Input(i, "value") for i in component_ids])
        def update_url_state(**kwargs):

            state = urlencode(kwargs, doseq=True)

            return f"?{state}"

        @self.callback(
            Output("filter_results", "children"),
            [Input(i, "value") for i in component_ids],
        )
        @dash_kwarg([Input(i, "value") for i in component_ids])
        def filter_results(**kwargs):

            # Filtering by AND-ing conditions together

            data_sources_json_file = open("../shared_config/data_sources.json")
            source_series_list = json.load(data_sources_json_file)
            data_sources_json_file.close()

            forecast_series_dicts = {}

            for series_dict in source_series_list:
                try:
                    forecast_series_dicts[
                        series_dict["title"]
                    ] = get_forecast_data(series_dict["title"])
                except FileNotFoundError:
                    continue

            filters = {
                "name": match_names,
                "tags": match_tags,
                "methods": match_methods,
            }

            list_filter_matches = []

            for filter_key, filter_fn in filters.items():
                matched_series_names = filter_fn(
                    forecast_series_dicts, kwargs[filter_key]
                )
                list_filter_matches.append(matched_series_names)

            unique_series_titles = list(
                sorted(set.intersection(*list_filter_matches))
            )

            if len(unique_series_titles) > 0:

                results_list = []

                for item_title in unique_series_titles:
                    series_data = forecast_series_dicts[item_title]
                    url_title = urlencode({"title": item_title})
                    thumbnail_figure = get_thumbnail_figure(series_data)

                    results_list.append(
                        html.Div(
                            [
                                html.A(
                                    [
                                        html.H5(item_title),
                                        dcc.Graph(
                                            figure=thumbnail_figure,
                                            config={"displayModeBar": False},
                                        ),
                                    ],
                                    href=f"/series?{url_title}",
                                ),
                                html.Hr(),
                            ]
                        )
                    )

                results = [
                    html.P(
                        f"{len(unique_series_titles)} result{'s' if len(unique_series_titles) > 1 else ''} found"
                    ),
                    html.Div(results_list),
                ]
            else:
                results = [html.P("No results found")]

            return results
