import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import time
from urllib.parse import urlparse, parse_qsl
from dash.exceptions import PreventUpdate
import dash
import pickle
import json
from collections import defaultdict
import functools
import operator
from flask_caching import Cache

header = [
    html.A("Home", href="/"),
    html.Br(),
    html.A("Australian Economic", href="/series?tags=Australia,Economic"),
]

def parse_state(url):
    parse_result = urlparse(url)
    params = parse_qsl(parse_result.query)
    state = dict(params)
    return state

def get_forecast_plot_data(series_df, forecast_df):

    line_history = go.Scatter(
        x=series_df["date"],
        y=series_df["value"],
        name='Historical',
        mode='lines+markers',
    )

    forecast_error_x = list(forecast_df.index) + list(reversed(forecast_df.index))
    forecast_error_x = [x.to_pydatetime() for x in forecast_error_x]

    error_50 = go.Scatter(
        x=forecast_error_x,
        y=list(forecast_df['UB_50']) + list(reversed(forecast_df['LB_50'])),
        fill='tozeroy',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name="50% CI",
    )

    error_75 = go.Scatter(
        x=forecast_error_x,
        y=list(forecast_df['UB_75']) + list(reversed(forecast_df['LB_75'])),
        fill='tozeroy',
        fillcolor='rgba(0,176,246,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name="75% CI",
    )

    error_95 = go.Scatter(
        x=forecast_error_x,
        y=list(forecast_df['UB_95']) + list(reversed(forecast_df['LB_95'])),
        fill='tozeroy',
        fillcolor='rgba(231,107,243,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name="95% CI",
    )

    line_forecast = go.Scatter(
        x=forecast_df.index,
        y=forecast_df["FORECAST"],
        name='Forecast'
    )

    data = [line_history, error_95, error_75, error_50, line_forecast]

    return data


def get_thumbnail_figure(title):

    # Load from CSV
    start = time.time()

    f = open(f"forecasts/{title}.pkl", 'rb')
    data_dict = pickle.load(f)

    series_df = data_dict['series_df']
    forecast_df = data_dict['forecast_df']

    data = get_forecast_plot_data(series_df, forecast_df)

    layout = go.Layout(
        title=title + ' Forecast',

        height=720,

        xaxis=dict(

            fixedrange=True,

            type='date',

            range=[
                series_df['date'].iloc[-16].to_pydatetime(), # Recent point in history
                forecast_df.index[-1].to_pydatetime() # End of forecast range
            ],
        ),

        yaxis=dict(
            fixedrange=True,  # Will disable all zooming and movement controls if True
            autorange=True,
        ),

    )

    end = time.time()
    print(end - start)
    return go.Figure(data, layout)

def get_series_figure(title):

    # Load from CSV
    start = time.time()

    f = open(f"forecasts/{title}.pkl", 'rb')
    data_dict = pickle.load(f)

    series_df = data_dict['series_df']
    forecast_df = data_dict['forecast_df']

    data = get_forecast_plot_data(series_df, forecast_df)

    time_difference_forecast_to_start = forecast_df.index[-1].to_pydatetime() - \
                                        series_df['date'][0].to_pydatetime()

    layout = go.Layout(
        title=title + ' Forecast',

        height=720,

        xaxis=dict(

            fixedrange=True,

            type='date',

            range=[
                series_df['date'].iloc[-16].to_pydatetime(), # Recent point in history
                forecast_df.index[-1].to_pydatetime() # End of forecast range
            ],

            rangeselector=dict(
                buttons=list([
                    dict(count=5,
                         label='5y',
                         step='year',
                         stepmode='backward'),
                    dict(count=10,
                         label='10y',
                         step='year',
                         stepmode='backward'),
                    dict(
                        count=time_difference_forecast_to_start.days,
                        label="all",
                        step='day',
                        stepmode='backward')
                ])
            ),

            rangeslider=dict(
                visible=True,
                range=[series_df['date'][0].to_pydatetime(), forecast_df.index[-1].to_pydatetime()]
            ),

        ),

        yaxis=dict(
            fixedrange=True,  # Will disable all zooming and movement controls if True
            autorange=True,
        ),

    )

    end = time.time()
    print(end - start)
    return go.Figure(data, layout)


class Index(dash.Dash):

    def __init__(self, name, server, url_base_pathname):

        # Must initialise the parent class
        external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

        super().__init__(
            name=name, server=server, url_base_pathname=url_base_pathname,
            external_stylesheets=external_stylesheets
        )

        showcase_item_titles = [
            "Australian GDP",
            "Australian Underemployment"
        ]

        self.cache = Cache(self.server, config = {
                           'CACHE_TYPE': 'filesystem',
                           'CACHE_DIR': 'cache-directory'
                           })
        
        @self.cache.memoize(timeout=3600)
        def layout_func():

            showcase_list = []

            for item_title in showcase_item_titles:
                showcase_list.append(html.Div(html.A([dcc.Graph(id=item_title, figure=get_thumbnail_figure(item_title), config={'displayModeBar': False, 'staticPlot': True})], href=f"/series?title={item_title}"), className="six columns"))

            showcase_div = html.Div(showcase_list, className='row')

            return html.Div(header + [html.H3('Gallery')] + [showcase_div])

        self.layout = layout_func

class Series(dash.Dash):

    def __init__(self, name, server, url_base_pathname):

        # Must initialise the parent class
        external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

        super().__init__(
            name=name, server=server, url_base_pathname=url_base_pathname,
            external_stylesheets=external_stylesheets
        )

        self.layout = html.Div(header + [
            dcc.Location(id="url", refresh=False),
            html.Div(id="dynamic_content"),
        ])

        @self.callback(
            Output('dynamic_content', 'children'),
            [Input('url', 'href')])
        def display_value(value):
            # Put this in to avoid an Exception due to weird Location component
            # behaviour
            if value is None:
                raise PreventUpdate

            print(value)
            parse_result = parse_state(value)

            if 'title' in parse_result:

                series_graph = dcc.Graph(figure=get_series_figure(parse_result['title']), config = {'modeBarButtonsToRemove': ['sendDataToCloud', 'autoScale2d', 'hoverClosestCartesian',
                                                     'hoverCompareCartesian', 'lasso2d', 'select2d',
                                                     'toggleSpikelines'], "displaylogo": False})

                return [html.H3('Series'), series_graph]
            elif 'tags' in parse_result:

                # Parse the data source tags for the filter page
                # Build a reverse dictionary for the tags
                tag_dict = defaultdict(list)

                with open("data_sources.json") as data_sources_json_file:

                    data_sources_list = json.load(data_sources_json_file)

                    for data_source_dict in data_sources_list:

                        for tag in data_source_dict["tags"]:
                            tag_dict[tag].append(data_source_dict["title"])


                tag_matches = [tag_dict[tag] for tag in parse_result['tags'].split(",")]
                unique_series_titles = set(functools.reduce(operator.iconcat, tag_matches, []))

                if len(unique_series_titles) > 0:

                    results_list = []

                    for item_title in unique_series_titles:
                        results_list.append(
                            html.Div([html.H5(item_title),
                                html.A([dcc.Graph(id=item_title,
                                                                        figure=get_thumbnail_figure(item_title),
                                                                        config={'displayModeBar': False,
                                                                                'staticPlot': True}, className="six columns")],
                                                             href=f"/series?title={item_title}")],
                                                      className="row"))

                    results_div = html.Div(results_list)

                    return [html.H3('Search Results'), results_div]

                else:
                    return [html.H3('Search Results'), html.H5("None")]
            else:
                raise PreventUpdate
