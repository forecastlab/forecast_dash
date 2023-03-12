import os

import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State

from datetime import datetime
import humanize
import pytz

import pickle

from slugify import slugify

from datetime import datetime
from urllib.parse import urlencode

import dash_bootstrap_components as dbc
from dash import dcc, html

import humanize
import numpy as np
import pandas as pd
from frontmatter import Frontmatter
from util import (
    glob_re,
    watermark_information,
)
import plotly.express as px
import json

home_route = ("Business Forecast Lab", "/")

nav_routes = [
    ("Find a Series", "/search/"),
    ("Leaderboard", "/leaderboard/"),
    ("Blog", "/blog"),
    ("Methodology", "/methodology/"),
    ("About", "/about/"),
]


def header():
    return dbc.Navbar(
        dbc.Container(
            [
                dbc.Row(
                    [
                        # dbc.Col(html.Img(src="/assets/USYD_coa_reversed.png", height="30px")),
                        dbc.Col(
                            html.A(
                                [
                                    html.Img(
                                        src="/assets/USYD_coa_reversed.png",
                                        height="30px",
                                    )
                                ],
                                href="https://www.sydney.edu.au/business/",
                            )
                        ),
                        dbc.Col(
                            dbc.NavbarBrand(
                                "Forecast Lab",
                                className="ms-2",
                                href="/",
                                external_link=True,
                            )
                        ),
                    ],
                    align="center",
                    className="g-0",
                ),
                dbc.Col(
                    dbc.Row(
                        [
                            dbc.NavbarToggler(id="navbar-toggler"),
                            dbc.Collapse(
                                dbc.Nav(
                                    [
                                        dbc.NavItem(
                                            dbc.NavLink(
                                                x[0],
                                                href=x[1],
                                                external_link=True,
                                            )
                                        )
                                        for x in nav_routes
                                    ]
                                    + [
                                        dbc.NavItem(
                                            dbc.NavLink(
                                                html.I(
                                                    className="fab fa-github fa-lg"
                                                ),
                                                href="https://github.com/forecastlab/forecast_dash",
                                                external_link=True,
                                            )
                                        )
                                    ],
                                    # make sure nav takes up the full width for auto
                                    # margin to get applied
                                    className="w-100",
                                ),
                                id="navbar-collapse",
                                is_open=False,
                                navbar=True,
                            ),
                        ],
                        # the row should expand to fill the available horizontal space
                        className="flex-grow-1",
                    ),  # close row
                    lg="expand",
                ),  # close col
            ],
        ),  # close containter
        color="dark",
        dark=True,
    )


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
        class_name="bg-light rounded-3 px-3 pt-3 mb-3",
    )


def component_git_version():
    git_hash = ""
    git_shorthash = "Unknown commit"
    git_time = "00:00"
    git_author = "Unknown author"
    git_subject = ""

    # Get the current git status
    # %n: literal newline
    # %H: commit hash
    # %h: abbreviated commit hash
    # %ai: author date
    # %an: author name
    # %s: subject
    # Gotcha: The seemingly redundant --git-dir ../.git is a workaround for
    # docker container isolation: .git is exported to /.git as read-only volume.
    git_cmd = (
        'git --git-dir ../.git show --no-patch --format="%H%n%h%n%ai%n%an%n%s"'
    )
    git_output = os.popen(git_cmd).read().splitlines()

    if len(git_output) >= 5:
        git_hash = git_output[0]
        git_shorthash = git_output[1]
        git_time = git_output[2]

        natural_time = humanize.naturaltime(
            datetime.strptime(git_time, "%Y-%m-%d %H:%M:%S %z"),
            when=datetime.now(tz=pytz.timezone("Australia/Sydney")),
        )
    else:
        natural_time = "Error Loading Timestamp"

    github_home_url = "https://github.com/forecastlab/forecast_dash/"
    github_patch_url = github_home_url + "commit/" + git_hash

    return dbc.Col(
        [
            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            html.H5(
                                [
                                    "Version ",
                                    html.A(
                                        git_shorthash,
                                        href=github_patch_url,
                                        className="text-decoration-none",
                                    ),
                                ],
                                className="card-title",
                            ),
                            html.H6(
                                f"Committed {natural_time}",
                                className="card-subtitle mb-2 text-muted",
                            ),
                            html.P(
                                [
                                    html.A(
                                        "Development homepage",
                                        href=github_home_url,
                                        className="text-decoration-none",
                                    ),
                                    " on ",
                                    html.A(
                                        "GitHub",
                                        href="https://github.com/",
                                        className="text-decoration-none",
                                    ),
                                ]
                            ),
                        ]
                    )
                ],
                color="dark",
                outline=True,
            ),
        ]
    )


def footer():
    return dbc.Container(
        [
            dbc.Row(
                dbc.Col(html.Hr(style={"margin-top": "64px"}), lg=12),
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Nav(
                                [
                                    dbc.NavItem(
                                        dbc.NavLink(
                                            x[0], href=x[1], external_link=True
                                        )
                                    )
                                    for x in [home_route] + nav_routes
                                ],
                                vertical="md",
                            )
                        ],
                        lg=7,
                        style={"margin-bottom": "16px"},
                    ),
                    component_git_version(),
                ],
                style={"margin-bottom": "64px"},
            ),
        ]
    )


def markdown_layout(title, markdown_content):
    return html.Div(
        [
            dcc.Location(id="url", refresh=False),
            dbc.Container(
                [
                    breadcrumb_layout([("Home", "/"), (title, "")]),
                    dcc.Markdown(markdown_content),
                ]
            ),
        ]
    )


### model selection & visulisation related
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


def world_map_of_forecasts():
    with open("../shared_config/data_sources.json") as data_json:
        df = json.load(data_json)

    # Load the list of countries
    countries_dict = {}
    countries = pd.read_csv("../data/CountriesList.csv")
    for c in countries["Country"]:
        countries_dict[c] = {"Series": 0, "Titles": []}

    tags_lists = {}
    # Update dynamically with new countries
    for d in df:
        for tag in d["tags"]:
            if tag in countries_dict:
                countries_dict[tag]["Series"] += 1
                countries_dict[tag]["Titles"] += [d["title"]]

    # Basic cleaning for plotly
    country_data = pd.DataFrame.from_dict(countries_dict, orient="index")
    country_data["Country"] = country_data.index
    country_data["Titles"] = [
        "<br>".join(i) for i in country_data["Titles"].values
    ]
    country_data = country_data[country_data["Series"] > 0]
    country_data = country_data.merge(countries)

    # Plotly express is clean for this
    fig = px.choropleth(
        data_frame=country_data,
        locations="Code",
        color="Series",
        hover_name="Country",
        color_continuous_scale="burgyl",
        projection="natural earth",
        custom_data=["Country"],
    )

    fig.update_layout(
        coloraxis_showscale=False,
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Rockwell"),
        dragmode=False,
        margin={"l": 16, "r": 16, "t": 0, "b": 0},
    )

    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br><i>%{z} Series</i>"
    )

    return fig
