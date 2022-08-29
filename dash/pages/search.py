import json
import pickle
import re

from urllib.parse import urlencode

import dash_bootstrap_components as dbc
from dash import dcc, html, callback
from dash.exceptions import PreventUpdate

import dash
import pandas as pd


from common import breadcrumb_layout
from dash.dependencies import Input, Output, State
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
from slugify import slugify


def sort_filter_results(unique_series_titles, sort_by="a_z", **kwargs):

    # Load all the forecast data slows this down. Keep this for now, if you are going to use any sorting by MSE
    # for item_title in unique_series_titles:
    #     series_data = forecast_series_dicts[item_title]
    #     title = series_data["data_source_dict"]["title"]
    #     best_model = select_best_model(forecast_series_dicts[item_title])
    #     mse = series_data["all_forecasts"][best_model]["cv_score"]["MSE"]

    #     df.append([title] #, best_model, mse])

    df = pd.DataFrame(
        unique_series_titles, columns=["Title"]
    )  # , "BestModel", "MSE"])

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


dash.register_page(__name__, title="Find a Series")

component_ids = ["name"]  # , "tags", "methods"]

# load data source
with open("../shared_config/data_sources.json") as data_sources_json_file:
    series_list = json.load(data_sources_json_file)

with open("../shared_config/search_a_series.json") as searches_json_file:
    searchable_details = json.load(searches_json_file)

### result layout
def result_layout():
    return dbc.Container(
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
                                [
                                    dcc.Loading(html.Div(id="filter_results")),
                                ]
                            ),
                            dbc.Row(
                                [
                                    html.Button(
                                        "Load more",
                                        id="load_new_content",
                                        n_clicks=0,
                                        className="fill",
                                    ),
                                ]
                            ),
                        ],
                        lg=12,
                        sm=12,
                    ),
                ]
            ),
        ]
    )


### functions for filtering, result displaying
def filter_panel_children(params):
    children = [
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            html.H4("Filters"),
                            dbc.Label("Name", html_for="name"),
                            apply_default_value(params)(dcc.Dropdown)(
                                id="name",
                                clearable=True,
                                placeholder="Name of a series or method...",
                                value="",
                                # multi=True,
                                options=add_dropdown_search_options(),
                            ),
                            dbc.FormText("Type something in the box above"),
                        ],
                        className="mb-3",
                    )
                ),
            ],
        ),
    ]
    return children


def match_names(searchable_details, name_input):
    if not name_input or name_input == "":
        all_titles = [s["title"] for s in series_list]
        return set(all_titles)

    matched_series_names = []

    # name_terms = "|".join(name_input.split(" ")) # keep for later use.
    name_terms = name_input  # for single search
    name_terms = name_terms.replace("(", "\\(")
    name_terms = name_terms.replace(")", "\\)")

    for search_term, result_titles in searchable_details.items():

        # Search title
        re_results = re.search(
            name_terms,
            search_term,
            re.IGNORECASE,
        )
        if re_results is not None:
            matched_series_names += result_titles  # now a list

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


def add_dropdown_search_options():

    all_tags = []

    for series_dict in series_list:
        all_tags.extend(series_dict["tags"])

    # The code below has been disabled for now, until a work around is possible.
    all_tags = [
        {
            "label": tag,  # html.Div(
            #     [
            #         html.Div(className="fa fa-hashtag"),
            #         html.Div(tag, style={"font-size": 15, "padding-left": 10}),
            #     ],
            #     style={
            #         "display": "flex",
            #         "align-items": "center",
            #         "justify-content": "center",
            #     },
            # ),
            "value": tag,
        }
        for tag in sorted(set(all_tags))
    ]

    # Load methods
    stats = get_forecast_data("statistics")
    all_methods = [
        # {
        #     "label": html.Div(
        #         [
        #             html.Div(className="fa fa-wrench"),
        #             html.Div(
        #                 f"Winning Method - {method}",
        #                 style={"font-size": 15, "padding-left": 10},
        #             ),
        #         ],
        #         style={
        #             "display": "flex",
        #             "align-items": "center",
        #             "justify-content": "center",
        #         },
        #     ),
        #     "value": method,
        # }
        {"label": f"Winning Method - {method}", "value": method}
        for method in sorted(stats["models_used"])
    ]

    all_titles = []
    for series_dict in series_list:
        all_titles.append(series_dict["title"])
        try:
            all_titles.append(series_dict["short_title"])
        except:
            pass

    all_titles = [
        {
            "label": title,  # html.Div(
            #     [
            #         html.Div(className="fa fa-globe"),
            #         html.Div(
            #             title, style={"font-size": 15, "padding-left": 10}
            #         ),
            #     ],
            #     style={
            #         "display": "flex",
            #         "align-items": "center",
            #         "justify-content": "center",
            #     },
            # ),
            "value": title,
        }
        for title in all_titles
    ]

    all_options = all_tags + all_methods + all_titles
    all_options = sorted(
        all_options, key=lambda d: d["value"]
    )  # previous was 'label', can't order by html.Div

    return all_options


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

    return filter_panel_children(parse_result)


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
    inputs=[Input(i, "value") for i in component_ids]
    + [Input("results_sort_input", "value")]
    + [Input("load_new_content", "n_clicks")],
)
@dash_kwarg(
    [Input(i, "value") for i in component_ids]
    + [Input("results_sort_input", "value")]
    + [Input("load_new_content", "n_clicks")]
)
def filter_results(**kwargs):

    # Fix up name
    if type(kwargs["name"]) == list:
        kwargs["name"] = "".join(kwargs["name"])

    # Filtering by AND-ing conditions together

    forecast_series_dicts = {}

    n_clicks = kwargs["load_new_content"]

    filters = {
        "name": match_names,
        # "tags": match_tags,
        # "methods": match_methods,
    }

    list_filter_matches = []

    for filter_key, filter_fn in filters.items():
        matched_series_names = filter_fn(
            searchable_details, kwargs[filter_key]
        )
        list_filter_matches.append(matched_series_names)

    unique_series_titles = list(sorted(set.intersection(*list_filter_matches)))

    unique_series_titles = sort_filter_results(
        unique_series_titles,
        # forecast_series_dicts,
        sort_by=kwargs["results_sort_input"],
    )

    if len(unique_series_titles) > 0:

        n_series = len(unique_series_titles)

        results_list = []

        for item_title in unique_series_titles[
            0 : (30 + (n_clicks + 1) * 9)
        ]:  # show first thirty nine
            try:
                forecast_series_dicts[item_title] = get_forecast_data(
                    item_title
                )

                series_data = forecast_series_dicts[item_title]
                url_title = urlencode({"title": item_title})

                title = (
                    series_data["data_source_dict"]["short_title"]
                    if "short_title" in series_data["data_source_dict"]
                    else series_data["data_source_dict"]["title"]
                )

                try:

                    thumbnail_figure = open(
                        f"./../data/thumbnails/{slugify(item_title)}.pkl", "rb"
                    )
                    thumbnail_figure = pickle.load(thumbnail_figure)
                except Exception as e:
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
                        sm=12,
                        md=6,
                        lg=4,
                        xl=4,
                    ),
                )
            except FileNotFoundError:
                continue

        results = [
            html.P(f"{n_series} result{'s' if n_series > 1 else ''} found"),
            html.Div(dbc.Row(results_list)),
        ]
    else:
        results = [html.P("No results found")]

    return results


def make_card(item_title, url_title, thumbnail_figure, best_model):
    return dbc.Card(
        [
            html.A(
                [
                    dbc.CardImg(
                        src=thumbnail_figure,
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
                                # html.P(),
                                html.H6(
                                    f"{best_model}",
                                    className="card-text mt-auto",
                                    style={
                                        "color": "black",
                                        "font-weight": "italic",
                                        "text-align": "right",
                                    },
                                ),
                            ],
                            className="card-img-overlay d-flex flex-column justify-content",
                        ),
                    ),
                ],
                href=f"/series?{url_title}",
            )
        ]
    )


### final layout
def layout(name=None, tags=None, methods=None):
    return html.Div(
        [
            dcc.Location(id="url", refresh=False),
            result_layout(),
        ]
    )
