### Functions for model visualisation, selection, ...
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
# from common import BootstrapApp, header, breadcrumb_layout, footer
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

##### model selection
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

##### model visualisation
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

def get_forecast_data(title):
    f = open(f"../data/forecasts/{title}.pkl", "rb")
    data_dict = pickle.load(f)
    return data_dict

##### news
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

