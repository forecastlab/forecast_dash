import ast
import os
import re
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
