from datetime import datetime

import dash_bootstrap_components as dbc
from dash import dcc, html, callback
import dash
import dash_dangerously_set_inner_html

import humanize
from common import breadcrumb_layout
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import frontmatter
from util import (
    glob_re,
    location_ignore_null,
    parse_state,
)

markdown_extras = ["cuddled-lists"]

dash.register_page(
    __name__,
    title="Post",
)


def _post_layout():
    return dbc.Container(
        [
            breadcrumb_layout(
                [("Home", "/"), ("Blog", "/blog"), ("Post", "")]
            ),
            dbc.Row(dbc.Col(id="post", lg=12)),
        ],
        style={"margin-bottom": "64px"},
    )


### update web content
def _find_post_filename(url):
    parse_result = parse_state(url)

    if "title" in parse_result:
        title = parse_result["title"][0]

    try:
        return glob_re(f"{title}.md", "../blog")[0]  # filename
    except:
        raise PreventUpdate


def _load_blog_post(filename):
    blog_post = frontmatter.load("../blog/" + filename)
    blog_post["filename"] = filename.split(".md")[0]
    return blog_post


@callback(Output("breadcrumb", "children"), [Input("url", "href")])
@location_ignore_null([Input("url", "href")], location_id="url")
def update_breadcrumb(url):
    filename = _find_post_filename(url)
    blog_post = _load_blog_post(filename)

    return blog_post["title"]


def _post_title(blog_post):
    return html.A(
        html.H2(blog_post["title"]),
        href=f"/blog?post={blog_post['filename']}",
        id=blog_post["filename"],
    )


def _post_author(blog_post):
    return html.P(
        [
            " by ",
            blog_post["author"],
            ", ",
            humanize.naturaltime(
                datetime.now()
                - datetime.strptime(blog_post["date"], "%Y-%m-%d")
            ),
        ],
        className="subtitle mt-0 text-muted small",
    )


def _post_content(blog_post):
    return (
        dash_dangerously_set_inner_html.DangerouslySetInnerHTML(str(blog_post))
        if "type" in blog_post.keys() and blog_post["type"] == "html"
        else dcc.Markdown(blog_post)
    )


@callback(Output("post", "children"), [Input("url", "href")])
@location_ignore_null([Input("url", "href")], location_id="url")
def update_content(url):
    filename = _find_post_filename(url)
    blog_post = _load_blog_post(filename)

    return [
        _post_title(blog_post),
        html.Hr(),
        _post_author(blog_post),
        _post_content(blog_post),
    ]


def layout(title=None):
    return html.Div(
        [
            dcc.Location(id="url", refresh=False),
            _post_layout(),
        ]
    )
