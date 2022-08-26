import json
from urllib.parse import urlencode
import dash
from dash import Dash, html, dcc, Output, Input, State, callback
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

from util import (
    location_ignore_null,
    parse_state,
    apply_default_value,
    dash_kwarg,
)

from common import (
    get_forecast_data,
    select_best_model,
    get_plot_shapes,
    watermark_information,
    breadcrumb_layout,
)

dash.register_page(__name__)

### plot function just for the double series case here
def get_forecast_plot_data(series_df, forecast_df, color_index=0, yaxis='y'):

    alpha = 0.3

    colors_group = [
        [
            f'rgba(255, 0, 0, 1)',
            f'rgba(226, 87, 78, {alpha})',
            f'rgba(234, 130, 112, {alpha})',
            f'rgba(243, 179, 160, {alpha})',
        ],
        [
            f'rgba(0, 0, 255, 1)',
            f'rgba(0, 191, 255, {alpha})',
            f'rgba(65, 105, 225, {alpha})',
            f'rgba(0, 0, 139, {alpha})',

        ]
    ]
    # in an order of history/CI50/CI75/CI95

    # Plot series history
    line_history = dict(
        type="scatter",
        x=series_df.index,
        y=series_df["value"],
        name="Historical",
        mode="lines+markers",
        line=dict(color=colors_group[color_index][0]),
        yaxis=yaxis,
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
        fillcolor=colors_group[color_index][1],
        line=dict(color="rgba(255,255,255,0)"),
        name="50% CI",
        yaxis=yaxis,
    )

    # Plot CI75
    error_75 = dict(
        type="scatter",
        x=forecast_error_x,
        y=list(forecast_df["UB_75"]) + list(reversed(forecast_df["LB_75"])),
        fill="tozeroy",
        fillcolor=colors_group[color_index][2],
        line=dict(color="rgba(255,255,255,0)"),
        name="75% CI",
        yaxis=yaxis,
    )

    # Plot CI95
    error_95 = dict(
        type="scatter",
        x=forecast_error_x,
        y=list(forecast_df["UB_95"]) + list(reversed(forecast_df["LB_95"])),
        fill="tozeroy",
        fillcolor=colors_group[color_index][3],
        line=dict(color="rgba(255,255,255,0)"),
        name="95% CI",
        yaxis=yaxis,
    )

    # Plot forecast
    line_forecast = dict(
        type="scatter",
        x=forecast_df.index,
        y=forecast_df["forecast"],
        name="Forecast",
        mode="lines",
        line=dict(color=colors_group[color_index][0], dash="2px"),
        yaxis=yaxis,
    )

    data = [error_95, error_75, error_50, line_forecast, line_history]

    return data


### format title
def _format_figure_title(title, color):
    title_color = f'''<b><span style="color:{color}">{title}</span></b>'''
    return f'''<a href="/series?title={title}">{title_color}</a>'''


def get_series_figure(data_dict, model_name, data_dict2, model_name2, twin_axes=False):
    model_name = model_name.replace('WIN-', '')
    model_name2 = model_name2.replace('WIN-', '')

    series2_yaxis = 'y2' if twin_axes else 'y'

    watermark_config = watermark_information()

    series_df = data_dict["downloaded_dict"]["series_df"]
    forecast_df = data_dict["all_forecasts"][model_name]["forecast_df"]
    series_df2 = data_dict2["downloaded_dict"]["series_df"]
    forecast_df2 = data_dict2["all_forecasts"][model_name2]["forecast_df"]

    data = get_forecast_plot_data(series_df, forecast_df)
    data += get_forecast_plot_data(series_df2, forecast_df2, color_index=1, yaxis=series2_yaxis)
    shapes = get_plot_shapes(series_df, forecast_df)

    time_difference_forecast_to_start = (
        forecast_df.index[-1].to_pydatetime()
        - series_df.index[0].to_pydatetime()
    )

    title1 = (
        data_dict["data_source_dict"]["short_title"]
        if "short_title" in data_dict["data_source_dict"]
        else data_dict["data_source_dict"]["title"]
    )

    title2 = (
        data_dict2["data_source_dict"]["short_title"]
        if "short_title" in data_dict2["data_source_dict"]
        else data_dict2["data_source_dict"]["title"]
    )

    layout = dict(
        title=f'''{_format_figure_title(title1, "red")} VS {_format_figure_title(title2, "blue")}''',
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

    ### add twin axes
    if twin_axes:
        layout['yaxis2'] = {
            'anchor': 'x',
            'overlaying': 'y',
            'side': 'right',
        }

    return dict(data=data, layout=layout)

### functions for select title and tags
def get_unique_tags(title_tags, sort=True):
    '''
    title_tags is a dictionary storing series title and series tags pairs
    This function returns a list of unique tags
    '''
    all_tags = [tag for tags in title_tags.values() for tag in tags]
    all_tags = sorted(set(all_tags)) if sort else set(all_tags)
    return all_tags


def load_title_tag_pairs():
    '''
    Function to load unique tags, title-tag relationship, tag-title relationship
    '''
    with open("../shared_config/data_sources.json") as data_sources_json_file:
        series_list = json.load(data_sources_json_file)
        
    ### construct title_tag_relation
    title_tags = {series['title']: series['tags'] for series in series_list}
    
    unique_tags = get_unique_tags(title_tags, sort=True)
    unique_titles = sorted([series['title'] for series in series_list])
    
    ### construct tag_title_relation
    tag_titles = {}
    for tag in unique_tags:
        tag_titles[tag] = set([title for title in title_tags if tag in title_tags[title]])
    
    return unique_titles, unique_tags, title_tags, tag_titles


def get_filtered_titles(tags_selected, unique_titles, tag_titles):
    '''
    Filtering available series titles with selected tags
    '''
    if tags_selected is None: return unique_titles
    if len(tags_selected) == 0: return unique_titles

    filtered_options = None
    for tag in tags_selected:
        if filtered_options is None: filtered_options = tag_titles[tag]
        else: filtered_options = filtered_options & tag_titles[tag]
    return sorted(list(filtered_options))


unique_titles, unique_tags, title_tags, tag_titles = load_title_tag_pairs()


def get_series_methods(data_dict, CV_score_function="MSE"):
    '''
    load all methods for this series from a data_dict
    Indicating Winning methods
    '''
    all_models = set(data_dict['all_forecasts'].keys())
    best_model = select_best_model(data_dict, CV_score_function)

    other_models = sorted(all_models - {best_model})

    return [f'WIN-{best_model}'] + other_models

### layout function
def series_double_dropdown(id_prefix, text, series):

    ### series selection
    title_dropdown = dcc.Dropdown(
        unique_titles,
        value=series,
        placeholder="Select a Series",
        id=f'{id_prefix}-title-dropdown',
    )

    title_dropdown = dbc.Row(
        [
            dbc.Col(html.P("SERIES"), lg=2, sm=1,),
            dbc.Col(title_dropdown),
        ]
    )

    ### method selection
    method_dropdown = dcc.Dropdown(
        placeholder="Select a Method",
        id=f'{id_prefix}-method-dropdown',
    )

    method_dropdown = dbc.Row(
        [
            dbc.Col(html.P("METHOD"), lg=2, sm=1,),
            dbc.Col(method_dropdown),
        ]
    )

    return dbc.Col(
        [
            html.H4(text),
            dbc.Row(title_dropdown),
            dbc.Row(method_dropdown),
        ],
        align='center',
        style={"margin-top":"20px"}
        # lg=4,
        # sm=1,
    )


def series_selection_layout(series1=None, series2=None):
    return dbc.Col(
        [
            series_double_dropdown('series1', 'Series 1', series1),
            series_double_dropdown('series2', 'Series 2', series2),
        ],
        # justify="evenly",
    )


def twin_axes_switch_layout():
    return dbc.Checklist(
        options=[
            {"label": "Twin Axes", "value": 1},
        ],
        value=[0],
        id="twin_axes_input",
        switch=True,
    )


def series_link_layout(link_name, link_id, color):
    return html.A(
            html.Button(
                link_name, 
                style={
                    "background-color": color, 
                    "color":"white",
                    "border": "none",
                    "padding": "15px 32px",
                }
            ),
            href=None,
            id=link_id,
        )


def extra_info_row_layout():
    return dbc.Row(
        [
            dbc.Col([twin_axes_switch_layout()]),
            dbc.Col([series_link_layout("Link to Series 1", "series1-link", "#00A6FF")]),
            dbc.Col([series_link_layout("Link to Series 2", "series2-link", "#FF4200")]),
        ],
        style = {"margin-top" :"20px", "margin-bottom" :"20px"}
    )

### callbacks related
def _update_series_list(tags_selected):
    filtered_item = get_filtered_titles(tags_selected, unique_titles, tag_titles)
    if tags_selected is None: first_item = None
    elif len(tags_selected) == 0: first_item = None
    else: first_item = filtered_item[0] if len(filtered_item) > 0 else None

    return filtered_item, first_item

# @callback(
#     Output("series1-title-dropdown", "options"),
#     Output("series1-title-dropdown", "value"),
#     Input("series1-tag-dropdown", "value"),
#     prevent_initial_call=True,
# )
# def update_filtered_series1(tags_selected):
#     series_filtered = _update_series_list(tags_selected)
#     return series_filtered

# # for series2 - assure that two dropdown menu are not updated together
# @callback(
#     Output("series2-title-dropdown", "options"),
#     Output("series2-title-dropdown", "value"),
#     Input("series2-tag-dropdown", "value"),
#     prevent_initial_call=True,
# )
# #get_filtered_titles(tags_select, unique_titles, tag_titles)
# def update_filtered_series2(tags_selected):
#     series_filtered = _update_series_list(tags_selected)
#     return series_filtered


### update methods
@callback(
    Output("series1-method-dropdown", 'options'),
    Output("series1-method-dropdown", 'value'),
    Input("series1-title-dropdown", 'value'),
)
def update_series1_methods(series_title):
    if series_title is None:
        return [], None
    try:
        data_dict = get_forecast_data(series_title)
        methods = get_series_methods(data_dict)
        return methods, methods[0]
    except:
        return [], None


@callback(
    Output("series2-method-dropdown", 'options'),
    Output("series2-method-dropdown", 'value'),
    Input("series2-title-dropdown", 'value'),
)
def update_series2_methods(series_title):
    if series_title is None:
        return [], None
    try:
        data_dict = get_forecast_data(series_title)
        methods = get_series_methods(data_dict)
        return methods, methods[0]
    except:
        return [], None


### update links
@callback(
    Output("series1-link", 'href'),
    Input("series1-title-dropdown", 'value'),
)
def update_series1_methods(series_title):
    if series_title is None:
        return None
    return f'/series?title={series_title}'


### update links
@callback(
    Output("series2-link", 'href'),
    Input("series2-title-dropdown", 'value'),
)
def update_series1_methods(series_title):
    if series_title is None:
        return None
    return f'/series?title={series_title}'


### update url query
@callback(
    Output("URL", 'search'),
    Input("series1-title-dropdown", "value"),
    Input("series2-title-dropdown", "value"),
)
def update_url_query(title1, title2):
    # if title1 is None or title2 is None:
    #     raise PreventUpdate

    query = urlencode(
        [("title1", title1), ("title2", title2)],
    )

    return f'?{query}'


### Plotting callbacks
@callback(
    Output("series-graph", "children"),
    Input("series1-title-dropdown", 'value'),
    Input("series1-method-dropdown", "value"),
    Input("series2-title-dropdown", 'value'),
    Input("series2-method-dropdown", "value"),
    Input("twin_axes_input", "value"),
)
def update_series_graph(series_title1, method1, series_title2, method2, twin_axes_input):
    if not (series_title1 and series_title2 and method1 and method2):
        raise PreventUpdate

    twin_axes = False if twin_axes_input==[0] else True

    series_data_dict1 = get_forecast_data(series_title1)
    series_data_dict2 = get_forecast_data(series_title2)
    series_figure = get_series_figure(
        series_data_dict1, method1, series_data_dict2, method2, twin_axes,
    )

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
                filename=f"{series_title1} vs {series_title2}",
                format="svg",
                width=1024,
                height=768,
            ),
        },
    )

    return series_graph
    
    
### final layout
def layout(title1=None, title2=None,):
    return dbc.Container(
        [
            dcc.Location(id="URL", refresh=False), # not sure what happened here, lower case does not work...
            breadcrumb_layout([("Home", "/"), ("More Series", "")]),
            series_selection_layout(title1, title2),
            extra_info_row_layout(),
            dcc.Loading(dbc.Row([dbc.Col(id="series-graph")])),
        ]
    )

