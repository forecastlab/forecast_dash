import json
import re

from urllib.parse import urlencode

import dash_bootstrap_components as dbc
from dash import dcc, html, callback
import dash

from common import breadcrumb_layout
from dash.dependencies import Input, Output
from util import (
    location_ignore_null,
    parse_state,
    apply_default_value,
    dash_kwarg,
)
from common import (
    select_best_model,
    get_thumbnail_figure,
    get_forecast_data,
)

dash.register_page(
    __name__, 
    title="Find a Series"
)

component_ids = ["name", "tags", "methods"]

# load data source
with open("../shared_config/data_sources.json") as data_sources_json_file:
    series_list = json.load(data_sources_json_file)

### result layout
def result_layout():
    return dbc.Container(
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
    ])

### functions for filtering, result displaying
def filter_panel_children(params, tags, methods):
    children = [
        html.Div(
            [
                html.H4("Filters"),
                dbc.Label("Name", html_for="name"),
                apply_default_value(params)(dbc.Input)(
                    id="name",
                    placeholder="Name of a series...",
                    type="search",
                    value="",
                ),
                dbc.FormText("Type something in the box above"),
            ],
            className="mb-3",
        ),
        html.Div(
            [
                dbc.Label("Tags", html_for="tags"),
                apply_default_value(params)(dbc.Checklist)(
                    options=[{"label": t, "value": t} for t in tags],
                    value=[],
                    id="tags",
                ),
            ],
            className="mb-3",
        ),
        html.Div(
            [
                dbc.Label("Method", html_for="methods"),
                apply_default_value(params)(dbc.Checklist)(
                    options=[
                        {"label": m, "value": m} for m in methods
                    ],
                    value=[],
                    id="methods",
                ),
            ],
            className="mb-3",
        ),
    ]

    return children

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

@callback(
    Output("filter_panel", "children"), 
    Input("url", "href"),
)
@location_ignore_null([Input("url", "href")], "url")
def filter_panel(value):

    parse_result = parse_state(value)

    all_tags = []

    for series_dict in series_list:
        all_tags.extend(series_dict["tags"])

    all_tags = sorted(set(all_tags))

    # Dynamically load methods
    stats = get_forecast_data("statistics")
    all_methods = sorted(stats["models_used"])

    return filter_panel_children(parse_result, all_tags, all_methods)

@callback(
    Output("url", "search"),
    inputs=[Input(i, "value") for i in component_ids],
)
@dash_kwarg([Input(i, "value") for i in component_ids])
def update_url_state(**kwargs):

    state = urlencode(kwargs, doseq=True)

    return f"?{state}"

@callback(
    Output("filter_results", "children"),
    [Input(i, "value") for i in component_ids],
)
@dash_kwarg([Input(i, "value") for i in component_ids])
def filter_results(**kwargs):
    # Fix up name
    if type(kwargs["name"]) == list:
        kwargs["name"] = "".join(kwargs["name"])

    # Filtering by AND-ing conditions together

    forecast_series_dicts = {}

    for series_dict in series_list:
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

### final layout
def layout(name=None,tags=None,methods=None):
    return html.Div([
        dcc.Location(id="url", refresh=False),
        result_layout(),
    ])
