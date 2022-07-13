import json
import pickle
import re

from datetime import datetime
from functools import wraps
from urllib.parse import urlencode

import dash_bootstrap_components as dbc
from dash import dcc, html, callback
import dash

import humanize
import numpy as np
import pandas as pd
from dash import dash_table
from common import breadcrumb_layout
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from frontmatter import Frontmatter
from util import (
    glob_re,
    location_ignore_null,
    parse_state,
    apply_default_value,
    watermark_information,
    dash_kwarg,
)
from modelutil import (
    select_best_model,
    get_thumbnail_figure,
    get_forecast_data,
    component_news_4col,
    component_figs_2col,
)


import io
import base64

dash.register_page(
    __name__, 
    title="Leaderboard"
)

# load data source
with open("../shared_config/data_sources.json") as data_sources_json_file:
    series_list = json.load(data_sources_json_file)

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

def _get_scoring_functions():
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

    return CV_scoring_functions

def _make_model_select_options(CV_scoring_functions):
    CV_scoring_functions = list(set(CV_scoring_functions))
    all_CV_scoring_function_options = dict(
        zip(CV_scoring_functions, CV_scoring_functions)
    )
    model_select_options = [
        {"label": v, "value": k}
        for k, v in all_CV_scoring_function_options.items()
    ]

    return model_select_options
    
### layout functions
def _cv_table_layout():
    model_select_options = _make_model_select_options(_get_scoring_functions())
    return dbc.Container([
            breadcrumb_layout(
                [("Home", "/"), ("Leaderboard", "")]
            ),
            html.H2("Leaderboard"),
            dcc.Dropdown(
                id="leaderboard_CV_score_selector",
                clearable=False,
                options=model_select_options,
                value="MSE",
            ),
            dbc.Row([dbc.Col(id="leaderboard_CV_table")]),
    ])

@callback(
    Output("leaderboard_CV_table", "children"),
    Input("leaderboard_CV_score_selector", "value"),
)
@location_ignore_null([Input("url", "href")], location_id="url")
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

### final layout variable
layout = html.Div([
    dcc.Location(id="url", refresh=False),
    _cv_table_layout(),
])