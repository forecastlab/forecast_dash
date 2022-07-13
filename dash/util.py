import ast
import os
import re
from datetime import datetime
from functools import wraps
from urllib.parse import urlparse, parse_qs

from dash.exceptions import PreventUpdate


def glob_re(pattern, dir):
    return list(filter(re.compile(pattern).match, os.listdir(dir)))


def parse_state(url):
    parse_result = urlparse(url)
    return parse_qs(parse_result.query)


def location_ignore_null(inputs, location_id):
    def accept_func(func):
        @wraps(func)
        def wrapper(*args):
            input_names = [item.component_id for item in inputs]
            kwargs_dict = dict(zip(input_names, args))

            if kwargs_dict[location_id] is None:
                raise PreventUpdate

            return func(*args)

        return wrapper

    return accept_func


def apply_default_value(params):
    def wrapper(func):
        def apply_value(*args, **kwargs):
            if "id" in kwargs and kwargs["id"] in params:
                key = "value"
                try:
                    kwargs[key] = ast.literal_eval(params[kwargs["id"]])
                except Exception:
                    kwargs[key] = params[kwargs["id"]]

            return func(*args, **kwargs)

        return apply_value

    return wrapper


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

def dash_kwarg(inputs):
    def accept_func(func):
        @wraps(func)
        def wrapper(*args):
            input_names = [item.component_id for item in inputs]
            kwargs_dict = dict(zip(input_names, args))
            return func(**kwargs_dict)

        return wrapper

    return accept_func