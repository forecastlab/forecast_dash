from functools import wraps

import dash_bootstrap_components as dbc
from dash import dcc, html, callback, callback_context
import dash

import numpy as np
import pandas as pd
from dash import dash_table
from common import breadcrumb_layout
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from util import (
    location_ignore_null,
    parse_state,
)
from common import (
    select_best_model,
    get_series_figure,
    get_forecast_data,
)


import io
import base64

dash.register_page(__name__, title="Series")

# button style
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


def _forecast_info_layout():
    return dbc.Col(
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
    )


def _forecast_performance_layout():
    return dbc.Col(
        [
            dbc.Row(
                [dbc.Label("Model Cross Validation Scores")],
            ),
            dbc.Row(
                [
                    dbc.Checklist(
                        options=[
                            {"label": "Raw Scores", "value": 1},
                        ],
                        value=[0],
                        id="display_scores_input",
                        switch=True,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dcc.Loading(html.Div(id="CV_scores_table")),
                ]
            ),
        ],
        lg=6,
    )


### callback for updating `breadcrumb does not work...
### Use this function instead
def _get_series_title(title):
    try:
        series_data_dict = get_forecast_data(title)
    except:
        return "Series"

    return (
        series_data_dict["data_source_dict"]["short_title"]
        if "short_title" in series_data_dict["data_source_dict"]
        else series_data_dict["data_source_dict"]["title"]
    )


def _series_layout(title=None):
    breadcrumb_content = _get_series_title(title)
    return dbc.Container(
        [
            breadcrumb_layout([("Home", "/"), (breadcrumb_content, "")]),
            dcc.Loading(
                dbc.Row([dbc.Col(id="series_graph", lg=12)])
            ),  # Series figure
            dbc.Row(
                [
                    _forecast_info_layout(),
                    _forecast_performance_layout(),
                ]
            ),
        ]
    )


# ### functions for loading data etc.
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
                        series_data_dict["all_forecasts"][model]["cv_score"][
                            CV_score
                        ],
                        4,
                    )
            else:
                CV_score_df.at[model, CV_score] = np.round(
                    series_data_dict["all_forecasts"][model]["cv_score"][
                        CV_score
                    ],
                    4,
                )
    CV_score_df.sort_values(by=["MSE"], inplace=True)

    # Reorder columns so MSE is always first as this is the most popular scoring function for the conditional mean
    CV_score_df = CV_score_df[
        ["MSE"] + [x for x in CV_score_df.columns.tolist() if x != "MSE"]
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

    metadata_df["Forecast Date"] = series_data_dict["forecasted_at"].strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    metadata_df["Download Date"] = series_data_dict["downloaded_dict"][
        "downloaded_at"
    ].strftime("%Y-%m-%d %H:%M:%S")

    (
        metadata_df["Period Frequency"],
        metadata_df["Period Frequency Name"],
    ) = infer_frequency_from_forecast(series_data_dict, **kwargs)

    metadata_df["Data Source"] = series_data_dict["data_source_dict"]["url"]
    metadata_df["Forecast Source"] = "https://business-forecast-lab.com/"

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
    return "{:,.2f}".format(x)  # np.round(x, 2)  #


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


### functions for updating page content
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

# # @callback(Output("test-callback", "children"), inputs)
# @callback(Output("breadcrumb", "children"), inputs)
# @location_ignore_null(inputs, location_id="url")
# @series_input(inputs, location_id="url")
# def update_breadcrumb(series_data_dict):
#     return (
#         series_data_dict["data_source_dict"]["short_title"]
#         if "short_title" in series_data_dict["data_source_dict"]
#         else series_data_dict["data_source_dict"]["title"]
#     )
##### NOT SURE WHY THIS DOES NOT WORK #####
# Everything will crash using the callback here...
# But it works fine when updating some other elements...
# Use function `_get_series_title` instead


@callback(
    Output("series_graph", "children"),
    inputs + [Input("model_selector", "value")],
)
@location_ignore_null(inputs, location_id="url")
@series_input(inputs + [Input("model_selector", "value")], location_id="url")
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


@callback(
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

    all_methods_dict[best_model_name] = f"{best_model_name} - Best Model (MSE)"

    model_select_options = [
        {"label": v, "value": k} for k, v in all_methods_dict.items()
    ]

    return model_select_options, best_model_name


@callback(
    Output("meta_data_list", "children"),
    inputs + [Input("model_selector", "value")],
)
@location_ignore_null(inputs, location_id="url")
@series_input(inputs + [Input("model_selector", "value")], location_id="url")
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
                                series_data_dict["data_source_dict"]["url"],
                                href=series_data_dict["data_source_dict"][
                                    "url"
                                ],
                            )
                        ]
                    ),
                ]
            ),
        ]
    )


@callback(
    Output("CV_scores_table", "children"),
    inputs
    + [
        Input("model_selector", "value"),
        Input("display_scores_input", "value"),
    ],
)
@location_ignore_null(inputs, location_id="url")
@series_input(
    inputs
    + [
        Input("model_selector", "value"),
        Input("display_scores_input", "value"),
    ],
    location_id="url",
)
def update_CV_scores_table(series_data_dict, **kwargs):

    best_model_name = kwargs["model_selector"]
    relative_values = True if kwargs["display_scores_input"] == [0] else False

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
        rounded_dataframe = cv_table_by_benchmark(rounded_dataframe, **kwargs)

    rounded_dataframe["Model"] = rounded_dataframe.index
    # Reorder columns for presentation
    rounded_dataframe = rounded_dataframe[
        ["Model"] + rounded_dataframe.columns.tolist()[:-1]
    ]

    table = dash_table.DataTable(
        id="CV_scores_datatable",
        data=rounded_dataframe.to_dict("records"),
        columns=[{"name": i, "id": i} for i in rounded_dataframe.columns],
        sort_action="custom",
        sort_by=[],
        sort_mode="single",
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


@callback(
    Output("CV_scores_datatable", "data"),
    Input("CV_scores_datatable", "sort_by"),
    Input("CV_scores_datatable", "data"),
)
def update_sorting_for_table(sort_by, data):
    """This is an ugly hack, but it seems to work"""
    rounded_dataframe = pd.DataFrame(data)
    for col in rounded_dataframe.columns[1:]:  # first col is the model name
        rounded_dataframe[col] = rounded_dataframe[col].apply(
            lambda x: float(str(x).replace(",", ""))
        )

    if len(sort_by):
        dff = rounded_dataframe.sort_values(
            sort_by[0]["column_id"],
            ascending=sort_by[0]["direction"] == "asc",
            inplace=False,
        )
    else:
        # No sort is applied
        dff = rounded_dataframe

    print(dff)
    # Round and format so that trailing zeros still appear
    for col in dff.columns[1:]:  # first col is the model name
        dff[col] = dff[col].apply(cv_table_clean_notation)

    return dff.to_dict("records")


@callback(
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
    forecast_table = create_forecast_table_df(series_data_dict, **kwargs)
    CV_scores_table = create_CV_scores_table(series_data_dict)
    series_data = create_historical_series_table_df(series_data_dict, **kwargs)
    metadata_table = create_metadata_table(series_data_dict, **kwargs)

    xlsx_io = io.BytesIO()
    writer = pd.ExcelWriter(xlsx_io)

    forecast_table.to_excel(writer, sheet_name="forecasts", index=False)
    CV_scores_table.to_excel(writer, sheet_name="CV_scores")
    series_data.to_excel(writer, sheet_name="series_data", index=False)

    metadata_table.to_excel(writer, sheet_name="metadata")

    writer.save()
    xlsx_io.seek(0)
    media_type = (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    data = base64.b64encode(xlsx_io.read()).decode("utf-8")
    href_data_downloadable = f"data:{media_type};base64,{data}"
    return href_data_downloadable


def layout(title=None):
    return html.Div(
        [
            dcc.Location(id="url", refresh=False),
            _series_layout(title),
        ]
    )
