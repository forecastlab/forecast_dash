import plotly.graph_objects as go
from datetime import datetime
import json
import pickle
import os
import numpy as np
import pandas as pd
import base64


def watermark_information():
    current_date = datetime.today().strftime(
        "%Y/%m/%d"
    )  # Get the current date YYYY-MM-DD format for watermarking figures
    watermark_text = "https://business-forecast-lab.com - {}".format(
        current_date
    )
    watermark_font_size_dict = {
        12: 20,
        8: 15,
        6: 12,
        4: 10,
    }  # Size is based upon the number of columns in the row. based upon the lg argument in dcc.Col
    watermark_dict = {
        "text": watermark_text,
        "font_size": watermark_font_size_dict,
    }
    return watermark_dict


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


def get_static_thumbnail_figure(data_dict, lg=12):
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

    fig = go.Figure(dict(data=data, layout=layout))
    # fig.write_image(f"../data/thumbnails/{title}.png")
    img_bytes = fig.to_image(format="png")
    encoding = base64.b64encode(img_bytes).decode()
    img_b64 = "data:image/png;base64," + encoding
    print(f"Writing File: ../data/thumbnails/{title}.pkl")
    return img_b64


def get_forecast_data(title):
    f = open(f"../data/forecasts/{title}.pkl", "rb")
    data_dict = pickle.load(f)
    return data_dict

# Previous files are from pages.py
def generate_static_thumbnail(sources_path, download_path):
    if not os.path.exists(download_path):
        os.mkdir(download_path)

    data_sources_json_file = open(sources_path)
    series_list = json.load(data_sources_json_file)

    data_sources_json_file.close()

    forecast_series_dicts = {}
    series_titles = []
    for series_dict in series_list:
        try:
            forecast_series_dicts[series_dict["title"]] = get_forecast_data(
                series_dict["title"]
            )
            series_titles.append(series_dict["title"])
        except FileNotFoundError:
            continue

    for item_title in series_titles:

        series_data = forecast_series_dicts[item_title]
        img_b64 = get_static_thumbnail_figure(series_data)
        f = open(f"{download_path}/{item_title}.pkl", "wb")
        pickle.dump(img_b64, f)
        f.close()
