import json
import pickle
import re

from urllib.parse import urlencode

import dash_bootstrap_components as dbc
from dash import dcc, html, callback, ctx
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

component_ids = ["name", "show"]  # , "tags", "methods"]

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
                        [html.H4("Results")],
                    ),
                ],
            ),
            dbc.Row(
                [
                    dbc.Col(html.Div(id="filter_count"), lg={"size": 2}),
                    dbc.Col(
                        [
                            "Sort by",
                        ],
                        lg={"offset": 7, "size": 1},
                        className="text-end",
                    ),
                    dbc.Col(
                        [
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
                                ],
                                value="a_z",
                            ),
                        ],
                        lg={"size": 2},
                    ),
                ],
                className="mb-3",
                align="center",
            ),
            dbc.Row(
                [
                    dcc.Loading(html.Div(id="filter_results")),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    html.Div(
                        dbc.Button(
                            "Load more",
                            id="load_new_content",
                            n_clicks=0,
                            disabled=True,
                            size="lg",
                            color="light",
                        ),
                        className="d-grid gap-2 col-8 mx-auto",
                    )
                ],
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
                                multi=True,
                                options=add_dropdown_search_options(),
                            ),
                            dbc.FormText("Type something in the box above"),
                        ],
                        className="mb-3",
                    )
                ),
            ],
        ),
        apply_default_value(params, "data")(dcc.Store)(
            id="show", storage_type="memory", data=9
        )
        # apply_default_value(params, "value")(dcc.Input)(id="show", value=9)
    ]
    return children


def match_names(searchable_details, name_input):
    if not name_input or name_input == "":
        all_titles = [s["title"] for s in series_list]
        return set(all_titles)

    matched_series_names = []
    # print(name_input)
    # name_terms = "|".join(name_input.split(" ")) # keep for later use.
    name_terms = name_input  # for single search
    # name_terms = [name_term.replace("(", "\\(") for name_term in name_terms]
    # name_terms = [name_term.replace(")", "\\)") for name_term in name_terms]

    for _name in name_terms:
        result_titles = searchable_details[_name]
        if result_titles is not None:
            matched_series_names += result_titles  # now a list

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

    # all_tags = []

    # for series_dict in series_list:
    #     all_tags.extend(series_dict["tags"])

    # all_tags = sorted(set(all_tags))

    # # Dynamically load methods
    # stats = get_forecast_data("statistics")
    # all_methods = sorted(stats["models_used"])

    return filter_panel_children(parse_result)


@callback(
    Output("url", "search"),
    # inputs=[Input(i, "value") for i in component_ids],
    Input("name", "value"),
    Input("show", "data"),
)
# @dash_kwarg([Input(i, "value") for i in component_ids])
@dash_kwarg([Input("name", "value"), Input("show", "data")])
def update_url_state(**kwargs):
    state = urlencode(kwargs, doseq=True)

    return f"?{state}"


@callback(
    Output("filter_count", "children"),
    Output("filter_results", "children"),
    Output("load_new_content", "disabled"),
    Output("show", "data"),
    Input("name", "value"),
    Input("results_sort_input", "value"),
    Input("load_new_content", "n_clicks"),
    State("show", "data"),
)
@dash_kwarg(
    # [Input(i, "value") for i in component_ids]
    [Input("name", "value")]
    + [Input("results_sort_input", "value")]
    + [Input("load_new_content", "n_clicks")]
    + [State("show", "data")]
)
def filter_results(**kwargs):
    # Fix up name # keep as list now.
    # if type(kwargs["name"]) == list:
    #     kwargs["name"] = "".join(kwargs["name"])

    # Filtering by AND-ing conditions together

    forecast_series_dicts = {}

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

    # Check if load more button clicked (triggered)
    increment = 9
    show_num = kwargs["show"]
    if type(show_num) == list:
        show_num = show_num[0]
    show_num = int(show_num)

    if ctx.triggered_id == "load_new_content":
        if show_num < len(unique_series_titles):
            show_num += increment
    # elif ctx.triggered_id == "name" and show_num :
    #     print("setting to increment")
    #     show_num = increment

    if show_num < len(unique_series_titles):
        more_available = False
    else:
        more_available = True

    if len(unique_series_titles) > 0:
        n_series = len(unique_series_titles)

        results_list = []

        for item_title in unique_series_titles[0:show_num]:
            try:
                series_data = get_forecast_data(
                    item_title
                )
                if series_data == None:
                    raise FileNotFoundError

                forecast_series_dicts[item_title] = series_data

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
                        className="mb-3",
                    ),
                )
            except FileNotFoundError:
                continue

        counts = [f"{n_series} result{'s' if n_series > 1 else ''} found"]
        results = [
            html.Div(dbc.Row(results_list)),
        ]
    else:
        counts = ["No results found"]
        results = []

    return counts, results, more_available, show_num


def make_card(item_title, url_title, thumbnail_figure, best_model):
    return dbc.Card(
        [
            html.A(
                [
                    dbc.CardImg(
                        src=thumbnail_figure,
                        bottom=True,
                        style={
                            "opacity": 0.3,
                        },
                    ),
                    dbc.CardImgOverlay(
                        dbc.CardBody(
                            [
                                html.H4(
                                    item_title,
                                    className="card-title",
                                    style={
                                        "color": "black",
                                        "font-weight": "bold",
                                        "text-align": "center",
                                    },
                                ),
                                # html.P(),
                                # html.H6(
                                #     f"{best_model}",
                                #     className="card-text mt-auto",
                                #     style={
                                #         "color": "black",
                                #         "font-weight": "italic",
                                #         "text-align": "right",
                                #     },
                                # ),
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
def layout(**kwargs):
    return html.Div(
        [
            dcc.Location(id="url", refresh=False),
            result_layout(),
        ]
    )
