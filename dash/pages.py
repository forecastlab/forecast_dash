import json
import pickle
import re

from datetime import datetime
from functools import wraps
from urllib.parse import urlencode

import dash_bootstrap_components as dbc
from dash import dcc, html, callback_context

import humanize
import numpy as np
import pandas as pd
from dash import dash_table
from common import BootstrapApp, header, breadcrumb_layout, footer
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from frontmatter import Frontmatter
from util import (
    glob_re,
    location_ignore_null,
    parse_state,
    apply_default_value,
    watermark_information,
)

import io
import base64

from slugify import slugify


def dash_kwarg(inputs):
    def accept_func(func):
        @wraps(func)
        def wrapper(*args):
            input_names = [item.component_id for item in inputs]
            kwargs_dict = dict(zip(input_names, args))
            return func(**kwargs_dict)

        return wrapper

    return accept_func


def sort_filter_results(
    unique_series_titles, forecast_series_dicts, sort_by="a_z", *kwargs
):

    df = []

    for item_title in unique_series_titles:
        series_data = forecast_series_dicts[item_title]
        title = series_data["data_source_dict"]["title"]
        best_model = select_best_model(forecast_series_dicts[item_title])
        mse = series_data["all_forecasts"][best_model]["cv_score"]["MSE"]

        df.append([title, best_model, mse])

    df = pd.DataFrame(df, columns=["Title", "BestModel", "MSE"])

    if sort_by == "a_z":
        df.sort_values(by=["Title"], ascending=True, inplace=True)

    if sort_by == "z_a":
        df.sort_values(by=["Title"], ascending=False, inplace=True)

    # Keeping this for the moment, but not currently active
    # if sort_by == "mse_asc":
    #     df.sort_values(by=["MSE"], ascending=True, inplace=True)

    # if sort_by == "mse_desc":
    #     df.sort_values(by=["MSE"], ascending=False, inplace=True)

    sort_unique_series_title = df["Title"].values
    return sort_unique_series_title


def get_forecast_plot_data(series_df, forecast_df):

    # Plot series history
    line_history = dict(
        type="scatter",
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

    # Plot CI50
    error_50 = dict(
        type="scatter",
        x=forecast_error_x,
        y=list(forecast_df["UB_50"]) + list(reversed(forecast_df["LB_50"])),
        fill="tozeroy",
        fillcolor="rgb(226, 87, 78)",
        line=dict(color="rgba(255,255,255,0)"),
        name="50% CI",
    )

    # Plot CI75
    error_75 = dict(
        type="scatter",
        x=forecast_error_x,
        y=list(forecast_df["UB_75"]) + list(reversed(forecast_df["LB_75"])),
        fill="tozeroy",
        fillcolor="rgb(234, 130, 112)",
        line=dict(color="rgba(255,255,255,0)"),
        name="75% CI",
    )

    # Plot CI95
    error_95 = dict(
        type="scatter",
        x=forecast_error_x,
        y=list(forecast_df["UB_95"]) + list(reversed(forecast_df["LB_95"])),
        fill="tozeroy",
        fillcolor="rgb(243, 179, 160)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% CI",
    )

    # Plot forecast
    line_forecast = dict(
        type="scatter",
        x=forecast_df.index,
        y=forecast_df["forecast"],
        name="Forecast",
        mode="lines",
        line=dict(color="rgb(0,0,0)", dash="2px"),
    )

    data = [error_95, error_75, error_50, line_forecast, line_history]

    return data


def get_plot_shapes(series_df, forecast_df):

    shapes = [
        {
            "type": "rect",
            # x-reference is assigned to the x-values
            "xref": "x",
            "x0": series_df.index[0],
            "x1": series_df.index[-1],
            # y-reference is assigned to the plot paper [0,1]
            "yref": "paper",
            "y0": 0,
            "y1": 1,
            "fillcolor": "rgb(229, 236, 245)",
            "line": {"width": 0},
            "layer": "below",
        },
        {
            "type": "rect",
            # x-reference is assigned to the x-values
            "xref": "x",
            "x0": forecast_df.index[0],
            "x1": forecast_df.index[-1],
            # y-reference is assigned to the plot paper [0,1]
            "yref": "paper",
            "y0": 0,
            "y1": 1,
            "fillcolor": "rgb(206, 212, 220)",
            "line": {"width": 0},
            "layer": "below",
        },
    ]

    return shapes


def select_best_model(data_dict, CV_score_function="MSE"):
    # use the MSE as the default scoring function for identifying the best model.
    # Extract ( model_name, cv_score ) for each model.
    all_models = []
    all_cv_scores = []
    for model_name, forecast_dict in data_dict["all_forecasts"].items():
        if forecast_dict:
            all_models.append(model_name)
            if (
                forecast_dict["state"] == "OK"
                and type(forecast_dict["cv_score"]) == dict
            ):
                all_cv_scores.append(
                    forecast_dict["cv_score"][CV_score_function]
                )
            else:
                all_cv_scores.append(forecast_dict["cv_score"])

    # Select the best model.
    model_name = all_models[np.argmin(all_cv_scores)]

    return model_name


def get_thumbnail_figure(data_dict, lg=12):
    watermark_config = (
        watermark_information()
    )  # Grab the watermark text and fontsize information

    model_name = select_best_model(data_dict)
    series_df = data_dict["downloaded_dict"]["series_df"].iloc[-16:, :]
    forecast_df = data_dict["all_forecasts"][model_name]["forecast_df"]

    data = get_forecast_plot_data(series_df, forecast_df)
    shapes = get_plot_shapes(
        data_dict["downloaded_dict"]["series_df"], forecast_df
    )

    title = (
        data_dict["data_source_dict"]["short_title"]
        if "short_title" in data_dict["data_source_dict"]
        else data_dict["data_source_dict"]["title"]
    )

    layout = dict(
        title={"text": title, "xanchor": "auto", "x": 0.5},
        height=480,
        showlegend=False,
        xaxis=dict(
            fixedrange=True,
            range=[series_df.index[0], forecast_df.index[-1]],
            gridcolor="rgb(255,255,255)",
        ),
        yaxis=dict(fixedrange=True, gridcolor="rgb(255,255,255)"),
        shapes=shapes,
        margin={"l": 30, "r": 0, "t": 30},
        annotations=[
            dict(
                name="watermark",
                text=watermark_config["text"],
                opacity=0.2,
                font=dict(
                    color="black", size=watermark_config["font_size"][lg]
                ),
                xref="paper",
                yref="paper",
                x=0.025,  # x axis location relative to bottom left hand corner between (0,1)
                y=0.025,  # y axis location relative to bottom left hand corner between (0,1)
                showarrow=False,
            )
        ],
    )

    return dict(data=data, layout=layout)


def get_series_figure(data_dict, model_name):
    watermark_config = watermark_information()

    series_df = data_dict["downloaded_dict"]["series_df"]
    forecast_df = data_dict["all_forecasts"][model_name]["forecast_df"]

    data = get_forecast_plot_data(series_df, forecast_df)
    shapes = get_plot_shapes(series_df, forecast_df)

    time_difference_forecast_to_start = (
        forecast_df.index[-1].to_pydatetime()
        - series_df.index[0].to_pydatetime()
    )

    title = (
        data_dict["data_source_dict"]["short_title"]
        if "short_title" in data_dict["data_source_dict"]
        else data_dict["data_source_dict"]["title"]
    )

    layout = dict(
        title=title,
        height=720,
        xaxis=dict(
            fixedrange=True,
            type="date",
            gridcolor="rgb(255,255,255)",
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
            gridcolor="rgb(255,255,255)",
        ),
        annotations=[
            dict(
                name="watermark",
                text=watermark_config["text"],
                opacity=0.2,
                font=dict(
                    color="black", size=watermark_config["font_size"][12]
                ),
                xref="paper",
                yref="paper",
                x=0.025,  # x axis location relative to bottom left hand corner between (0,1)
                y=0.025,  # y axis location relative to bottom left hand corner between (0,1)
                showarrow=False,
            )
        ],
        shapes=shapes,
        modebar={"color": "rgba(0,0,0,1)"},
    )

    return dict(data=data, layout=layout)


def get_forecast_data(title):
    title = slugify(title)
    f = open(f"../data/forecasts/{title}.pkl", "rb")
    data_dict = pickle.load(f)
    return data_dict


def component_figs_2col(row_title, series_titles):

    if len(series_titles) != 2:
        raise ValueError("series_titles must have 3 elements")

    return dbc.Row(
        [
            dbc.Col(
                [
                    html.H2(row_title),
                ],
                lg=12,
                className="text-center",
            ),
        ]
        + [
            dbc.Col(
                [
                    html.A(
                        [
                            dcc.Graph(
                                figure=get_thumbnail_figure(
                                    get_forecast_data(series_title), lg=6
                                ),
                                config={"displayModeBar": False},
                            )
                        ],
                        href=f"/series?{urlencode({'title': series_title})}",
                    )
                ],
                lg=6,
            )
            for series_title in series_titles
        ]
    )


def component_figs_3col(row_title, series_titles):

    if len(series_titles) != 3:
        raise ValueError("series_titles must have 3 elements")

    return dbc.Row(
        [
            dbc.Col(
                [
                    html.H3(row_title, style={"text-align": "center"}),
                ],
                lg=12,
            ),
        ]
        + [
            dbc.Col(
                [
                    html.A(
                        [
                            dcc.Graph(
                                figure=get_thumbnail_figure(
                                    get_forecast_data(series_title), lg=4
                                ),
                                config={"displayModeBar": False},
                            )
                        ],
                        href=f"/series?{urlencode({'title': series_title})}",
                    )
                ],
                lg=4,
            )
            for series_title in series_titles
        ]
    )


def component_news_4col():

    filenames = glob_re(r".*.md", "../blog")

    blog_posts = []

    for filename in filenames:
        fm_dict = Frontmatter.read_file("../blog/" + filename)
        fm_dict["filename"] = filename.split(".md")[0]
        blog_posts.append(fm_dict)

    # Sort by date
    blog_posts = sorted(
        blog_posts, key=lambda x: x["attributes"]["date"], reverse=True
    )

    body = []

    for i in range(min(len(blog_posts), 5)):
        blog_post = blog_posts[i]
        blog_timedelta = humanize.naturaltime(
            datetime.now()
            - datetime.strptime(blog_post["attributes"]["date"], "%Y-%m-%d")
        )
        body.extend(
            [
                html.Div(
                    blog_timedelta, className="subtitle mt-0 text-muted small"
                ),
                html.A(
                    html.P(blog_post["attributes"]["title"], className="lead"),
                    href=f"/blog/post?title={blog_post['filename']}",
                    className="text-decoration-none",
                ),
            ]
        )

    return dbc.Col(
        [html.H3("Latest News")]
        + body
        + [
            html.A(
                html.P("View all posts"),
                href="/blog",
                className="text-decoration-none",
            )
        ],
        lg=4,
    )


def component_leaderboard_4col(series_list):

    leaderboard_counts = get_leaderboard_df(series_list).iloc[:10, :]

    body = []

    for index, row in leaderboard_counts.iterrows():
        body.append(
            html.Li(
                index,
                className="lead",
            )
        )

    return dbc.Col(
        [
            html.H3("Leaderboard"),
            html.P(
                "Ranked by number of times each method was selected as the best performer",
                className="subtitle text-muted",
            ),
            html.Ol(body),
            html.A(
                html.P("View full leaderboard"),
                href="/leaderboard",
            ),
        ],
        lg=4,
    )


class Index(BootstrapApp):
    def setup(self):

        self.title = "Business Forecast Lab"

        data_sources_json_file = open("../shared_config/data_sources.json")
        series_list = json.load(data_sources_json_file)
        data_sources_json_file.close()

        feature_series_title = "Australian Inflation (CPI)"

        def layout_func():

            return html.Div(
                header()
                + [
                    dcc.Location(id="url", refresh=False),
                    # Mission Statement
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
                    # Main Body
                    dbc.Container(
                        [
                            # Row 1 - Featured and Latest News
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H3(
                                                "Featured Series",
                                                style={"text-align": "center"},
                                            ),
                                            html.A(
                                                [
                                                    dcc.Graph(
                                                        figure=get_thumbnail_figure(
                                                            get_forecast_data(
                                                                feature_series_title
                                                            ),
                                                            lg=8,
                                                        ),
                                                        config={
                                                            "displayModeBar": False
                                                        },
                                                    )
                                                ],
                                                href=f"/series?{urlencode({'title': feature_series_title})}",
                                            ),
                                        ],
                                        lg=8,
                                    ),
                                    component_news_4col(),
                                ],
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H2(
                                                "US Recovery from COVID-19",
                                            ),
                                            html.A(
                                                html.H4(
                                                    "View all US forecasts"
                                                ),
                                                href="/search/?name=&tags=US",
                                                className="text-decoration-none",
                                            ),
                                        ],
                                        lg=4,
                                        className="text-center align-self-center",
                                    ),
                                    dbc.Col(
                                        [
                                            html.A(
                                                [
                                                    dcc.Graph(
                                                        figure=get_thumbnail_figure(
                                                            get_forecast_data(
                                                                "US Unemployment"
                                                            ),
                                                            lg=8,
                                                        ),
                                                        config={
                                                            "displayModeBar": False
                                                        },
                                                    )
                                                ],
                                                href=f"/series?{urlencode({'title': 'US Unemployment'})}",
                                            ),
                                        ],
                                        lg=8,
                                    ),
                                ],
                                className="d-flex",
                            ),
                            # Row 3 - Leaderboard
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Div(
                                                dbc.Container(
                                                    dbc.Row(
                                                        dbc.Col(
                                                            [
                                                                html.A(
                                                                    [
                                                                        html.H1(
                                                                            "Leaderboard",
                                                                            className="display-4",
                                                                        ),
                                                                    ],
                                                                    href="/leaderboard/",
                                                                    className="text-decoration-none text-reset",
                                                                ),
                                                                html.H2(
                                                                    "We backtest every model on every series.",
                                                                    className="mb-3",
                                                                ),
                                                                html.H2(
                                                                    "Daily.",
                                                                    className="mb-3",
                                                                ),
                                                                html.A(
                                                                    html.H4(
                                                                        "Go to Leaderboard"
                                                                    ),
                                                                    href="/leaderboard/",
                                                                    className="text-decoration-none",
                                                                ),
                                                            ],
                                                            className="text-center",
                                                        ),
                                                    ),
                                                    className="px-4",
                                                ),
                                                className="bg-light rounded-3 py-5 mb-4",
                                            ),
                                        ],
                                        lg=12,
                                    )
                                ]
                            ),
                            # Row 5 - UK Snapshot
                            component_figs_2col(
                                "UK Snapshot",
                                [
                                    "UK Unemployment",
                                    "UK Inflation (RPI)",
                                ],
                            ),
                            # Bottom Row - links to all series
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Div(
                                                dbc.Container(
                                                    dbc.Row(
                                                        dbc.Col(
                                                            [
                                                                html.H2(
                                                                    "Looking for Something?",
                                                                    className="mb-3",
                                                                ),
                                                                html.A(
                                                                    html.H4(
                                                                        "Filter by name, country, tags and more!",
                                                                        className="mb-3",
                                                                    ),
                                                                    href="/search/",
                                                                    className="text-decoration-none",
                                                                ),
                                                            ],
                                                            className="text-center",
                                                        ),
                                                    ),
                                                    className="px-4",
                                                ),
                                                className="bg-light rounded-3 py-5 mb-4",
                                            ),
                                        ],
                                        lg=12,
                                    )
                                ]
                            ),
                        ]
                        + footer()
                    ),
                ]
            )

        self.layout = layout_func


class Series(BootstrapApp):
    def setup(self):
        white_button_style = {
            "background": "#fff",
            "backface-visibility": "hidden",
            "border-radius": ".375rem",
            "border-style": "solid",
            "border-width": ".1rem",  # .125rem
            "box-sizing": "border-box",
            "color": "#212121",
            "cursor": "pointer",
            "display": "inline-block",
            # "font-family": "Circular,Helvetica,sans-serif",
            "font-size": "1.125rem",
            "font-weight": "500",  # 700
            "letter-spacing": "-.01em",
            "line-height": "1.3",
            # "padding": ".875rem 1.125rem",
            "position": "relative",
            "text-align": "center",
            "text-decoration": "none",
            "transform": "translateZ(0) scale(1)",
            "transition": "transform .2s",
            "height": "40px",
            "width": "200px",
        }

        self.layout = html.Div(
            header()
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
                                        dbc.Label("Forecast Method"),
                                        dcc.Dropdown(
                                            id="model_selector",
                                            clearable=False,
                                        ),
                                        html.A(
                                            "Download Forecast Data",
                                            id="forecast_data_download_link",
                                            download="forecast_data.xlsx",
                                            href="",
                                            target="_blank",
                                        ),
                                        dcc.Loading(
                                            html.Div(
                                                id="meta_data_list",
                                                className="py-3",
                                            )
                                        ),
                                    ],
                                    lg=6,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Label(
                                                    "Model Cross Validation Scores"
                                                )
                                            ],
                                        ),
                                        dbc.Row(
                                            [
                                                html.Button(
                                                    "Relative Scores",
                                                    id="relative-val",
                                                    n_clicks=0,
                                                    style=white_button_style,
                                                ),
                                                html.Button(
                                                    "Raw Scores",
                                                    id="raw-val",
                                                    n_clicks=0,
                                                    style=white_button_style,
                                                ),
                                            ]
                                        ),
                                        dbc.Row(
                                            [
                                                dcc.Loading(
                                                    html.Div(
                                                        id="CV_scores_table"
                                                    )
                                                ),
                                            ]
                                        ),
                                    ],
                                    lg=6,
                                ),
                            ]
                        ),
                    ]
                    + footer()
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
            return (
                series_data_dict["data_source_dict"]["short_title"]
                if "short_title" in series_data_dict["data_source_dict"]
                else series_data_dict["data_source_dict"]["title"]
            )

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
                    "displayModeBar": True,
                    "toImageButtonOptions": dict(
                        filename=f"{model_name}",
                        format="svg",
                        width=1024,
                        height=768,
                    ),
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
            all_methods = list(series_data_dict["all_forecasts"].keys())

            all_methods_dict = dict(zip(all_methods, all_methods))

            all_methods_dict[
                best_model_name
            ] = f"{best_model_name} - Best Model (MSE)"

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

            return dbc.ListGroup(
                [
                    dbc.ListGroupItem(
                        [
                            html.H4("Model Details"),
                            html.P(
                                [
                                    html.P(model_name),
                                    html.P(model_description),
                                ]
                            ),
                        ]
                    ),
                    dbc.ListGroupItem(
                        [
                            html.H4("Forecast Updated At"),
                            html.P(
                                series_data_dict["forecasted_at"].strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                )
                            ),
                        ]
                    ),
                    dbc.ListGroupItem(
                        [
                            html.H4("Data Collected At"),
                            html.P(
                                series_data_dict["downloaded_dict"][
                                    "downloaded_at"
                                ].strftime("%Y-%m-%d %H:%M:%S")
                            ),
                        ]
                    ),
                    dbc.ListGroupItem(
                        [
                            html.H4("Data Source"),
                            html.P(
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

        def create_historical_series_table_df(series_data_dict, **kwargs):
            """
            Creates a Pandas DataFrame containing the historical time series data
            """

            dataframe = pd.DataFrame(
                series_data_dict["downloaded_dict"]["series_df"]["value"]
            )
            dataframe["date"] = dataframe.index.strftime("%Y-%m-%d %H:%M:%S")
            dataframe = dataframe[
                ["date"] + dataframe.columns.tolist()[:-1]
            ]  # reorder columns so the date is first

            return dataframe

        def create_forecast_table_df(series_data_dict, **kwargs):
            """
            Creates a Pandas DataFrame containing the point forecasts and confidence interval forecasts
            for a given forecast model
            """
            model_name = kwargs["model_selector"]

            forecast_dataframe = series_data_dict["all_forecasts"][model_name][
                "forecast_df"
            ]

            column_name_map = {"forecast": "value"}

            forecast_dataframe = forecast_dataframe.rename(
                column_name_map, axis=1
            ).round(4)
            forecast_dataframe["date"] = forecast_dataframe.index.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            forecast_dataframe["model"] = model_name
            forecast_dataframe = forecast_dataframe[
                ["date", "model"] + forecast_dataframe.columns.tolist()[:-2]
            ]  # reorder columns so the date and model columns first

            return forecast_dataframe

        def create_CV_scores_table(series_data_dict):
            """
            Creates a Pandas DataFrame containing the cross-validation scores for all scoring functions for all forecast models
            """

            # grab the list of all possible CV scores
            df_column_labels = []
            for model in series_data_dict["all_forecasts"].keys():
                df_column_labels = [
                    x
                    for x in series_data_dict["all_forecasts"][model][
                        "cv_score"
                    ].keys()
                    if "Winkler" not in x
                ]
            df_column_labels = df_column_labels + [
                "95% Winkler"
            ]  # only report the Winkler score for the 95% CI in the series page
            df_column_labels = list(set(df_column_labels))

            CV_score_df = pd.DataFrame(
                columns=df_column_labels,
                index=list(series_data_dict["all_forecasts"].keys()),
            )
            for model in list(series_data_dict["all_forecasts"].keys()):
                for CV_score in list(
                    series_data_dict["all_forecasts"][model]["cv_score"].keys()
                ):
                    if "Winkler" in CV_score:
                        if (
                            "95" in CV_score
                        ):  # only present the 95% CV score in the table
                            CV_score_df.at[model, "95% Winkler"] = np.round(
                                series_data_dict["all_forecasts"][model][
                                    "cv_score"
                                ][CV_score],
                                4,
                            )
                    else:
                        CV_score_df.at[model, CV_score] = np.round(
                            series_data_dict["all_forecasts"][model][
                                "cv_score"
                            ][CV_score],
                            4,
                        )
            CV_score_df.sort_values(by=["MSE"], inplace=True)

            # Reorder columns so MSE is always first as this is the most popular scoring function for the conditional mean
            CV_score_df = CV_score_df[
                ["MSE"]
                + [x for x in CV_score_df.columns.tolist() if x != "MSE"]
            ]

            return CV_score_df

        def infer_frequency_from_forecast(series_data_dict, **kwargs):
            """
            Not an efficient way of getting the periods frequency but can work for now.

            """
            model_name = kwargs["model_selector"]

            forecast_dataframe = series_data_dict["all_forecasts"][model_name][
                "forecast_df"
            ]

            # Select the number of forecasts made
            forecasts_len = len(forecast_dataframe.index) - 1

            # Map days to frequency reverse
            forecast_len_map_numbers = {13: 52, 18: 12, 8: 4, 4: 1}
            forecast_len_map_names = {
                13: "Weekly",
                18: "Monthly",
                8: "Quarterly",
                4: "Yearly",
            }

            return (
                forecast_len_map_numbers[forecasts_len],
                forecast_len_map_names[forecasts_len],
            )

        def create_metadata_table(series_data_dict, **kwargs):

            metadata_df = {}

            metadata_df["Forecast Date"] = series_data_dict[
                "forecasted_at"
            ].strftime("%Y-%m-%d %H:%M:%S")
            metadata_df["Download Date"] = series_data_dict["downloaded_dict"][
                "downloaded_at"
            ].strftime("%Y-%m-%d %H:%M:%S")

            metadata_df["Version"] = series_data_dict["downloaded_dict"][
                "data_version"
            ]

            (
                metadata_df["Period Frequency"],
                metadata_df["Period Frequency Name"],
            ) = infer_frequency_from_forecast(series_data_dict, **kwargs)

            metadata_df["Data Source"] = series_data_dict["data_source_dict"][
                "url"
            ]
            metadata_df[
                "Forecast Source"
            ] = "https://business-forecast-lab.com/"

            metadata_df = pd.DataFrame.from_dict(metadata_df, orient="index")
            metadata_df.columns = ["Value"]

            return metadata_df

        # Format to clean string so tables don't have very large numbers. anything larger than 4 characters can go to scientific notation.
        # If using 2 decimal places, add this to the map function.
        def cv_table_clean_notation(x):
            # return (
            #     "{:,.2f}".format(x)
            #     if len(str(int(x))) <= 4
            #     else "{:,.2e}".format(x)
            # )
            return "{:,.2f}".format(x)

        def cv_table_by_benchmark(df, benchmark_col=None, **kwargs):
            """
            Sets the validation scores relative to the best score or a column of your choosing.
            """
            if benchmark_col is None:
                benchmark_col = kwargs["model_selector"]
            for col in df.columns:
                x = df[col]
                bm = x[benchmark_col]
                x = x / bm
                df[col] = x
            return df

        @self.callback(
            Output("CV_scores_table", "children"),
            inputs
            + [
                Input("model_selector", "value"),
                Input("relative-val", "n_clicks"),
                Input("raw-val", "n_clicks"),
            ],
        )
        @location_ignore_null(inputs, location_id="url")
        @series_input(
            inputs
            + [
                Input("model_selector", "value"),
                Input("relative-val", "n_clicks"),
                Input("raw-val", "n_clicks"),
            ],
            location_id="url",
        )
        def update_CV_scores_table(series_data_dict, **kwargs):

            best_model_name = kwargs["model_selector"]
            changed_id = [p["prop_id"] for p in callback_context.triggered][0]
            relative_values = False if "raw-val" in changed_id else True

            # Dictionary of scoring function descriptions to display when hovering over in the CV scores table.
            tooltip_header_text = {
                "MSE": "Mean Squared Error of the point forecasts",
                "MASE": "Mean Absolute Scaled Error of the point forecasts",
                "95% Winkler": "Winkler score for the 95% prediction interval",
                "wQL25": "The Weighted Quantile Loss metric for the 25% quantile",
                "WAPE": "Weighted Absolute Percentage Error of the point forecasts",
                "SMAPE": "Symmetric Mean Absolute Percentage Error of the point forecasts",
            }

            dataframe = create_CV_scores_table(series_data_dict)
            rounded_dataframe = dataframe.copy()

            if relative_values:
                rounded_dataframe = cv_table_by_benchmark(
                    rounded_dataframe, **kwargs
                )
            # Round and format so that trailing zeros still appear
            for col in rounded_dataframe.columns:
                rounded_dataframe[col] = rounded_dataframe[col].apply(
                    cv_table_clean_notation
                )
            rounded_dataframe["Model"] = rounded_dataframe.index
            # Reorder columns for presentation
            rounded_dataframe = rounded_dataframe[
                ["Model"] + rounded_dataframe.columns.tolist()[:-1]
            ]

            table = dash_table.DataTable(
                id="CV_scores_datatable",
                data=rounded_dataframe.to_dict("records"),
                columns=[
                    {"name": i, "id": i} for i in rounded_dataframe.columns
                ],
                sort_action="native",
                # sort_mode="multi",
                style_cell={
                    "textAlign": "left",
                    "fontSize": 16,
                    "font-family": "helvetica",
                },
                tooltip_header=tooltip_header_text,
                tooltip_duration=None,  # Force the tooltip display for as long as the users cursor is over the header
                style_cell_conditional=[
                    {"if": {"column_id": "Model"}, "textAlign": "left"}
                ],
                style_header={"fontWeight": "bold", "fontSize": 18},
                style_header_conditional=[
                    {"if": {"column_id": col}, "textDecoration": "underline"}
                    for col in rounded_dataframe.columns
                    if col != "Model"
                ],  # underline headers associated with tooltips
                # style_data_conditional=[
                #     {
                #         "if": {
                #             "filter_query": "{{Model}} = {}".format(
                #                 best_model_name
                #             ), "column_id": "Model"
                #         },
                #         "backgroundColor": "powderblue",
                #         "color": "white",
                #     },
                # ],
                style_as_list_view=True,
            )
            return table

        @self.callback(
            Output("forecast_data_download_link", "href"),
            inputs + [Input("model_selector", "value")],
            prevent_initial_call=True,
        )
        @location_ignore_null(inputs, location_id="url")
        @series_input(
            inputs
            + [
                Input("model_selector", "value"),
            ],
            location_id="url",
        )
        def download_excel(series_data_dict, **kwargs):
            # Create DFs
            forecast_table = create_forecast_table_df(
                series_data_dict, **kwargs
            )
            CV_scores_table = create_CV_scores_table(series_data_dict)
            series_data = create_historical_series_table_df(
                series_data_dict, **kwargs
            )
            metadata_table = create_metadata_table(series_data_dict, **kwargs)

            xlsx_io = io.BytesIO()
            writer = pd.ExcelWriter(xlsx_io)

            forecast_table.to_excel(
                writer, sheet_name="forecasts", index=False
            )
            CV_scores_table.to_excel(writer, sheet_name="CV_scores")
            series_data.to_excel(writer, sheet_name="series_data", index=False)

            metadata_table.to_excel(writer, sheet_name="metadata")

            writer.save()
            xlsx_io.seek(0)
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            data = base64.b64encode(xlsx_io.read()).decode("utf-8")
            href_data_downloadable = f"data:{media_type};base64,{data}"
            return href_data_downloadable


def get_leaderboard_df(series_list, CV_score_function="MSE"):
    try:
        stats = get_forecast_data("statistics")
        all_methods = stats["models_used"]
    except FileNotFoundError:
        all_methods = []

    forecast_series_dicts = {}

    for series_dict in series_list:
        try:
            forecast_series_dicts[series_dict["title"]] = get_forecast_data(
                series_dict["title"]
            )
        except FileNotFoundError:
            continue

    chosen_methods = []

    for series_title, forecast_data in forecast_series_dicts.items():
        model_name = select_best_model(
            forecast_data, CV_score_function=CV_score_function
        )
        chosen_methods.append(model_name)

    stats_raw = pd.DataFrame({"Method": chosen_methods})

    unchosen_methods = list(set(all_methods) - set(chosen_methods))
    unchosen_counts = pd.Series(
        data=np.zeros(len(unchosen_methods)),
        index=unchosen_methods,
        name="Total Wins",
    )

    counts = pd.DataFrame(
        stats_raw["Method"]
        .value_counts()
        .rename("Total Wins")
        .append(unchosen_counts)
    )

    return counts


class Leaderboard(BootstrapApp):
    def setup(self):

        data_sources_json_file = open("../shared_config/data_sources.json")
        series_list = json.load(data_sources_json_file)
        data_sources_json_file.close()

        # grab the CV scoring function choices to populate dash dcc.Dropdown menu
        CV_scoring_functions = []
        for series_dict in series_list:
            try:
                forecast_series = get_forecast_data(series_dict["title"])

                for model in forecast_series["all_forecasts"].keys():
                    CV_scoring_functions += list(
                        forecast_series["all_forecasts"][model][
                            "cv_score"
                        ].keys()
                    )
            except:
                pass

        CV_scoring_functions = list(set(CV_scoring_functions))
        all_CV_scoring_function_options = dict(
            zip(CV_scoring_functions, CV_scoring_functions)
        )
        model_select_options = [
            {"label": v, "value": k}
            for k, v in all_CV_scoring_function_options.items()
        ]

        inputs = [Input("url", "href")]

        self.layout = html.Div(
            header()
            + [
                dcc.Location(id="url", refresh=False),
                dbc.Container(
                    [
                        breadcrumb_layout(
                            [("Home", "/"), (f"{self.title}", "")]
                        ),
                        html.H2(self.title),
                        dcc.Dropdown(
                            id="leaderboard_CV_score_selector",
                            clearable=False,
                            options=model_select_options,
                            value="MSE",
                        ),
                        dbc.Row([dbc.Col(id="leaderboard_CV_table")]),
                    ]
                    + footer()
                ),
            ]
        )

        @self.callback(
            Output("leaderboard_CV_table", "children"),
            Input("leaderboard_CV_score_selector", "value"),
        )
        @location_ignore_null(inputs, location_id="url")
        def update_leaderboard_df(CV_score):
            """
            construct the best model leaderboard based upon user selected scoring function
            """
            # Build leaderboard with chosen CV scoring function
            counts = get_leaderboard_df(series_list, CV_score)

            win_proportion = (
                counts["Total Wins"] / counts["Total Wins"].sum() * 100
            ).apply(lambda x: f" ({x:.2f}%)")

            counts["Total Wins"] = (
                counts["Total Wins"].apply(lambda x: f"{x:.0f}")
                + win_proportion
            )

            table = dbc.Table.from_dataframe(
                counts, index=True, index_label="Method"
            )

            # Apply URLS to index
            for row in table.children[1].children:
                state = urlencode(
                    {"methods": [row.children[0].children]}, doseq=True
                )
                row.children[0].children = html.A(
                    row.children[0].children, href=f"/search/?{state}"
                )

            return table


def match_names(forecast_dicts, name_input):
    if not name_input or name_input == "":
        return set(forecast_dicts.keys())

    matched_series_names = []

    name_terms = "|".join(name_input.split(" "))

    for series_title, forecast_dict in forecast_dicts.items():

        # Search title
        re_results = re.search(
            name_terms,
            forecast_dict["data_source_dict"]["title"],
            re.IGNORECASE,
        )
        if re_results is not None:
            matched_series_names.append(series_title)

        # Search short_title
        if "short_title" in forecast_dict["data_source_dict"]:
            re_results = re.search(
                name_terms,
                forecast_dict["data_source_dict"]["short_title"],
                re.IGNORECASE,
            )
            if re_results is not None:
                matched_series_names.append(series_title)

        # Search tags
        series_tags = " ".join(forecast_dict["data_source_dict"]["tags"])
        re_results = re.search(
            name_terms,
            series_tags,
            re.IGNORECASE,
        )
        # print(f"{series_title}, and tages :{series_tags}")
        if re_results is not None:
            matched_series_names.append(series_title)

        # Search methods
        re_results = re.search(
            name_terms,
            select_best_model(forecast_dict),
            re.IGNORECASE,
        )
        if re_results is not None:
            matched_series_names.append(series_title)

    return set(matched_series_names)


class Search(BootstrapApp):
    def setup(self):

        self.config.suppress_callback_exceptions = True

        # Dynamically load tags
        data_sources_json_file = open("../shared_config/data_sources.json")
        self.series_list = json.load(data_sources_json_file)
        data_sources_json_file.close()

        self.layout = html.Div(
            header()
            + [
                dcc.Location(id="url", refresh=False),
                dbc.Container(
                    [
                        breadcrumb_layout([("Home", "/"), ("Filter", "")]),
                        dbc.Row(dbc.Col(id="filter_panel", lg=12, sm=12)),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [html.H4("Results")],
                                                ),
                                                dbc.Col(
                                                    [
                                                        "Sort by:",
                                                        dcc.Dropdown(
                                                            id="results_sort_input",
                                                            clearable=False,
                                                            options=[
                                                                {
                                                                    "label": "A-Z Ascending",
                                                                    "value": "a_z",
                                                                },
                                                                {
                                                                    "label": "A-Z Descending",
                                                                    "value": "z_a",
                                                                },
                                                                # {'label': 'MSE Ascending', 'value': 'mse_asc'},
                                                                # {'label': 'MSE Descending', 'value': 'mse_desc'},
                                                            ],
                                                            value="a_z",
                                                        ),
                                                    ],
                                                    align="left",
                                                    lg=2,
                                                    sm=1,
                                                ),
                                            ],
                                            className="flex-grow-1",
                                        ),
                                        dbc.Row(
                                            dcc.Loading(
                                                html.Div(id="filter_results")
                                            ),
                                        ),
                                    ],
                                    lg=12,
                                    sm=12,
                                ),
                            ]
                        ),
                    ]
                    + footer(),
                ),
            ]
        )

        def filter_panel_children(params, tags, methods):
            children = [
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    html.H4("Filters"),
                                    dbc.Label("Name", html_for="name"),
                                    apply_default_value(params)(dbc.Input)(
                                        id="name",
                                        placeholder="Name of a series or method...",
                                        type="search",
                                        value="",
                                    ),
                                    dbc.FormText(
                                        "Type something in the box above"
                                    ),
                                ],
                                className="mb-3",
                            )
                        ),
                    ],
                ),
                # dbc.Col(
                #     html.Div(
                #         [
                #             dbc.Label("Tags", html_for="tags"),
                #             apply_default_value(params)(dbc.Checklist)(
                #                 options=[
                #                     {"label": t, "value": t} for t in tags[:1]
                #                 ],
                #                 value=[],
                #                 id="tags",
                #             ),
                #         ],
                #         className="mb-3",
                #     )
                # # ),
                # dbc.Row(
                #     html.Div(
                #         [
                #             # dbc.Label("Method", html_for="methods"),
                #             # apply_default_value(params)(dbc.Checklist)(
                #             dbc.Checklist(
                #                 options=[
                #                     {"label": "Sort by MSE", "value": "mse"}
                #                 ],
                #                 value=[],
                #                 id="sortingoption",
                #             ),
                #         ],
                #         className="mb-3",
                #     )
                # ),
            ]

            return children

        component_ids = ["name"]  # , "tags", "methods"]

        @self.callback(
            Output("filter_panel", "children"), [Input("url", "href")]
        )
        @location_ignore_null([Input("url", "href")], "url")
        def filter_panel(value):

            parse_result = parse_state(value)

            all_tags = []

            for series_dict in self.series_list:
                all_tags.extend(series_dict["tags"])

            all_tags = sorted(set(all_tags))

            # Dynamically load methods
            stats = get_forecast_data("statistics")
            all_methods = sorted(stats["models_used"])

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
            inputs=[Input(i, "value") for i in component_ids]
            + [Input("results_sort_input", "value")],
        )
        @dash_kwarg(
            [Input(i, "value") for i in component_ids]
            + [Input("results_sort_input", "value")]
        )
        def filter_results(**kwargs):

            # Fix up name
            if type(kwargs["name"]) == list:
                kwargs["name"] = "".join(kwargs["name"])

            # Filtering by AND-ing conditions together

            forecast_series_dicts = {}

            for series_dict in self.series_list:
                try:
                    forecast_series_dicts[
                        series_dict["title"]
                    ] = get_forecast_data(series_dict["title"])
                except FileNotFoundError:
                    continue

            filters = {
                "name": match_names,
                # "tags": match_tags,
                # "methods": match_methods,
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

            unique_series_titles = sort_filter_results(
                unique_series_titles,
                forecast_series_dicts,
                sort_by=kwargs["results_sort_input"],
            )

            if len(unique_series_titles) > 0:

                def make_card(
                    item_title, url_title, thumbnail_figure, best_model
                ):
                    return dbc.Card(
                        [
                            html.A(
                                [
                                    dbc.CardImg(
                                        src=thumbnail_figure,
                                        # dcc.Graph(
                                        #     figure=thumbnail_figure,
                                        #     config={"displayModeBar": False},
                                        # ),
                                        # src = "https://dash-bootstrap-components.opensource.faculty.ai/static/images/placeholder286x180.png",
                                        top=True,
                                        style={
                                            "opacity": 0.3,
                                        },
                                    ),
                                    dbc.CardImgOverlay(
                                        dbc.CardBody(
                                            [
                                                html.H4(
                                                    item_title,
                                                    className="card-title align-item-start",
                                                    style={
                                                        "color": "black",
                                                        "font-weight": "bold",
                                                        "text-align": "center",
                                                    },
                                                ),
                                                html.P(),
                                                html.P(
                                                    f"{best_model}",
                                                    className="card-text align-item-end",
                                                    style={
                                                        "color": "black",
                                                        "font-weight": "italic",
                                                        "text-align": "right",
                                                    },
                                                ),
                                            ],
                                            className="card-img-overlay d-flex flex-column justify-content-end",
                                        ),
                                    ),
                                ],
                                href=f"/series?{url_title}",
                            )
                        ]
                    )

                n_series = len(unique_series_titles)

                results_list = []

                for item_title in unique_series_titles:

                    series_data = forecast_series_dicts[item_title]
                    url_title = urlencode({"title": item_title})

                    title = (
                        series_data["data_source_dict"]["short_title"]
                        if "short_title" in series_data["data_source_dict"]
                        else series_data["data_source_dict"]["title"]
                    )

                    try:
                        thumbnail_figure = open(
                            f"../data/thumbnails/{item_title}.pkl", "rb"
                        )
                        thumbnail_figure = pickle.load(thumbnail_figure)
                    except:
                        # if no thumbnail image generated
                        thumbnail_figure = "https://dash-bootstrap-components.opensource.faculty.ai/static/images/placeholder286x180.png"

                    best_model = select_best_model(
                        forecast_series_dicts[item_title]
                    )

                    results_list.append(
                        dbc.Col(
                            make_card(
                                title, url_title, thumbnail_figure, best_model
                            ),
                            sm=3,
                        ),
                    )

                results = [
                    html.P(
                        f"{n_series} result{'s' if n_series > 1 else ''} found"
                    ),
                    html.Div(dbc.Row(results_list)),
                ]
            else:
                results = [html.P("No results found")]

            return results
