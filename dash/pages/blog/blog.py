from bs4 import BeautifulSoup
import textwrap

import math

from datetime import datetime

import dash_bootstrap_components as dbc
from dash import dcc, html, callback
import dash

import markdown2
import humanize
from common import breadcrumb_layout
from dash.dependencies import Input, Output
from frontmatter import Frontmatter
from util import (
    glob_re,
    location_ignore_null,
    parse_state,
)

markdown_extras = ["cuddled-lists"]

dash.register_page(__name__, title="Blog", path="/blog/")


def _blog_layout():
    return dbc.Container(
        [
            breadcrumb_layout([("Home", "/"), ("Blog", "")]),
            dbc.Row(
                [
                    dbc.Col([html.H1("Recent posts"), html.Hr()]),
                ]
            ),
            html.Div(id="body"),
        ],
        style={"margin-bottom": "64px"},
    )


### work out the parameters
def _find_page_number(value):
    parse_result = parse_state(value)

    if "page" not in parse_result:
        parse_result["page"] = ["1"]

    return int(parse_result["page"][0])  # page_int


def _collect_blog_posts():
    filenames = glob_re(r".*.md", "../blog")
    n_posts = len(filenames)

    blog_posts = []

    for filename in filenames:
        fm_dict = Frontmatter.read_file("../blog/" + filename)
        fm_dict["filename"] = filename.split(".md")[0]
        blog_posts.append(fm_dict)

    # Sort by date
    blog_posts = sorted(
        blog_posts, key=lambda x: x["attributes"]["date"], reverse=True
    )

    return blog_posts, n_posts


# format post review
def _post_review_title(blog_post):
    return html.A(
        html.H2(
            blog_post["attributes"]["title"],
            style={"padding-top": "8px"},
        ),
        href=f"/blog/post?title={blog_post['filename']}",
        id=blog_post["filename"],
    )


def _post_review_author(blog_post):
    return html.P(
        [
            " by ",
            blog_post["attributes"]["author"],
            ", ",
            humanize.naturaltime(
                datetime.now()
                - datetime.strptime(
                    blog_post["attributes"]["date"],
                    "%Y-%m-%d",
                )
            ),
        ],
        className="subtitle mt-0 text-muted small",
    )


def _post_review_abstract(blog_post):
    # load blog into bs4 format
    if (
        "type" in blog_post["attributes"]
        and blog_post["attributes"]["type"] == "html"
    ):
        body_html = blog_post["body"]
    else:
        body_html = markdown2.markdown(
            blog_post["body"], extras=markdown_extras
        )
    soup = BeautifulSoup(body_html, "html.parser")
    preview = textwrap.shorten(
        soup.find("p").get_text(), 280, placeholder="..."
    )

    return html.Div(preview, style={"padding-bottom": "8px"})


def _post_review_readmore(blog_post):
    return html.A(
        html.P(
            html.Strong(
                "Read more",
                className="text-left",
            ),
            style={"padding-bottom": "24px"},
        ),
        href=f"/blog/post?title={blog_post['filename']}",
    )


# add bottom navigation
# Previous | Page X of Y | Earlier
def _navigation_previous(page_int, n_pages):
    previous_link = (
        html.A(
            html.P("< Previous Posts"),
            id="previous_link",
            href=f"?page={page_int+1}",
            className="text-left",
        )
        if page_int < n_pages
        else []
    )

    return dbc.Col(
        previous_link,
        lg=2,
    )


def _navigation_pagecount(page_int, n_pages):
    return dbc.Col(
        html.P(
            f"Page {page_int} of {n_pages}",
            className="text-center",
        ),
        lg=4,
    )


def _navigation_earlier(page_int):
    earlier_link = (
        html.A(
            html.P("Earlier Posts >"),
            id="previous_link",
            href=f"?page={page_int-1}",
            className="text-right",
        )
        if page_int > 1
        else []
    )

    return dbc.Col(
        earlier_link,
        lg=2,
    )


def _render_post_reviews(value, n_posts_per_page=5):  # value from url
    page_int = _find_page_number(value)
    blog_posts, n_posts = _collect_blog_posts()

    start = (page_int - 1) * n_posts_per_page
    end = min((page_int) * n_posts_per_page, n_posts)

    body = []

    ### post review
    for i in range(start, end):
        blog_post = blog_posts[i]

        body.append(
            dbc.Row(
                dbc.Col(
                    [
                        _post_review_title(blog_post),
                        _post_review_author(blog_post),
                        _post_review_abstract(blog_post),
                        _post_review_readmore(blog_post),
                        html.Hr(),
                    ],
                    lg=8,
                )
            )
        )

    ### bottom navigation
    n_pages = math.ceil(n_posts / n_posts_per_page)

    body.append(
        dbc.Row(
            [
                _navigation_previous(page_int, n_pages),
                _navigation_pagecount(page_int, n_pages),
                _navigation_earlier(page_int),
            ]
        )
    )

    return body


@callback(Output("body", "children"), [Input("url", "href")])
@location_ignore_null([Input("url", "href")], "url")
def body(value):
    return _render_post_reviews(value)


### final layout function
def layout(page=None, post=None):
    return html.Div(
        [
            dcc.Location(id="url", refresh=False),
            _blog_layout(),
        ]
    )
