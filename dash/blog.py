import textwrap
from datetime import datetime

import dash_bootstrap_components as dbc
from dash import dcc, html
import dash_dangerously_set_inner_html
import humanize
import markdown2
from common import BootstrapApp, header, breadcrumb_layout, footer
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from frontmatter import Frontmatter
from multipage import Route, MultiPageApp
from util import glob_re, location_ignore_null, parse_state

import math

markdown_extras = ["cuddled-lists"]


class Blog(BootstrapApp):
    def setup(self):

        self.layout = html.Div(
            header()
            + [
                dcc.Location(id="url", refresh=False),
                dbc.Container(
                    [
                        breadcrumb_layout([("Home", "/"), ("Blog", "")]),
                        dbc.Row(
                            [
                                dbc.Col([html.H1("Recent posts"), html.Hr()]),
                            ]
                        ),
                        html.Div(id="body"),
                    ]
                    + footer(),
                    style={"margin-bottom": "64px"},
                ),
            ]
        )

        @self.callback(Output("body", "children"), [Input("url", "href")])
        @location_ignore_null([Input("url", "href")], "url")
        def body(value):

            # Find page number
            parse_result = parse_state(value)

            if "page" not in parse_result:
                parse_result["page"] = ["1"]

            page_int = int(parse_result["page"][0])

            # Load blog posts
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

            from bs4 import BeautifulSoup

            # Render post previews
            # For each blog post:
            # - Convert body to HTML
            # - Find <p> tags using bs4 and convert to text
            # - Truncate where neccesary

            n_posts_per_page = 5

            start = (page_int - 1) * n_posts_per_page
            end = min((page_int) * n_posts_per_page, n_posts)

            body = []

            for i in range(start, end):
                blog_post = blog_posts[i]

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
                    soup.find('p').get_text(), 280, placeholder="..."
                )

                body.append(
                    dbc.Row(
                        dbc.Col(
                            [
                                html.A(
                                    html.H2(
                                        blog_post["attributes"]["title"],
                                        style={"padding-top": "8px"},
                                    ),
                                    href=f"post?title={blog_post['filename']}",
                                    id=blog_post["filename"],
                                ),
                                html.P(
                                    [
                                        " by ",
                                        blog_post["attributes"]["author"],
                                        ", ",
                                        humanize.naturaltime(
                                            datetime.now()
                                            - datetime.strptime(
                                                blog_post["attributes"][
                                                    "date"
                                                ],
                                                "%Y-%m-%d",
                                            )
                                        ),
                                    ],
                                    className="subtitle mt-0 text-muted small",
                                ),
                                html.Div(
                                    preview, style={"padding-bottom": "8px"}
                                ),
                                html.A(
                                    html.P(
                                        html.Strong(
                                            "Read more",
                                            className="text-left",
                                        ),
                                        style={"padding-bottom": "24px"},
                                    ),
                                    href=f"post?title={blog_post['filename']}",
                                ),
                                html.Hr(),
                            ],
                            lg=8,
                        )
                    )
                )

            # Add bottom navigation
            # Previous | Page X of Y | Earlier

            n_pages = math.ceil(n_posts / n_posts_per_page)

            body.append(
                dbc.Row(
                    [
                        dbc.Col(
                            html.A(
                                html.P("< Previous Posts"),
                                id="previous_link",
                                href=f"?page={page_int+1}",
                                className="text-left",
                            )
                            if page_int < n_pages
                            else [],
                            lg=2,
                        ),
                        dbc.Col(
                            html.P(
                                f"Page {page_int} of {n_pages}",
                                className="text-center",
                            ),
                            lg=4,
                        ),
                        dbc.Col(
                            html.A(
                                html.P("Earlier Posts >"),
                                id="previous_link",
                                href=f"?page={page_int-1}",
                                className="text-right",
                            )
                            if page_int > 1
                            else [],
                            lg=2,
                        ),
                    ]
                )
            )

            return body


class Post(BootstrapApp):
    def setup(self):

        self.layout = html.Div(
            header()
            + [
                dcc.Location(id="url", refresh=False),
                dbc.Container(
                    [
                        breadcrumb_layout(
                            [("Home", "/"), ("Blog", "/blog"), ("Post", "")]
                        ),
                        dbc.Row(dbc.Col(id="post", lg=12)),
                    ]
                    + footer(),
                    style={"margin-bottom": "64px"},
                ),
            ]
        )

        inputs = [Input("url", "href")]

        @self.callback(Output("breadcrumb", "children"), inputs)
        @location_ignore_null(inputs, location_id="url")
        def update_breadcrumb(url):
            parse_result = parse_state(url)

            if "title" in parse_result:
                title = parse_result["title"][0]

            try:
                filename = glob_re(f"{title}.md", "../blog")[0]
            except:
                raise PreventUpdate

            fm_dict = Frontmatter.read_file("../blog/" + filename)
            fm_dict["filename"] = filename.split(".md")[0]

            return fm_dict["attributes"]["title"]

        @self.callback(Output("post", "children"), inputs)
        @location_ignore_null(inputs, location_id="url")
        def update_content(url):
            parse_result = parse_state(url)

            if "title" in parse_result:
                title = parse_result["title"][0]

            try:
                filename = glob_re(f"{title}.md", "../blog")[0]
            except:
                raise PreventUpdate

            blog_post = Frontmatter.read_file("../blog/" + filename)
            blog_post["filename"] = filename.split(".md")[0]

            return [
                html.A(
                    html.H2(blog_post["attributes"]["title"]),
                    href=f"/blog?post={blog_post['filename']}",
                    id=blog_post["filename"],
                ),
                html.Hr(),
                html.P(
                    [
                        " by ",
                        blog_post["attributes"]["author"],
                        ", ",
                        humanize.naturaltime(
                            datetime.now()
                            - datetime.strptime(
                                blog_post["attributes"]["date"], "%Y-%m-%d"
                            )
                        ),
                    ],
                    className="subtitle mt-0 text-muted small",
                ),
                dash_dangerously_set_inner_html.DangerouslySetInnerHTML(
                    blog_post["body"]
                )
                if "type" in blog_post["attributes"]
                and blog_post["attributes"]["type"] == "html"
                else dcc.Markdown(blog_post["body"]),
            ]


class BlogSection(MultiPageApp):
    def get_routes(self):
        return [
            Route(Blog, "Blog", "/"),
            Route(Post, "Post", "/post/"),
        ]
