import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from common import BootstrapApp, header, breadcrumb_layout, footer
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from frontmatter import Frontmatter
from multipage import Route, MultiPageApp
from util import glob_re, location_ignore_null, parse_state


class Blog(BootstrapApp):
    def setup(self):

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
            body.extend(
                [
                    html.A(
                        html.H1(blog_post["attributes"]["title"]),
                        href=f"post?title={blog_post['filename']}",
                        id=blog_post["filename"],
                    ),
                    html.Hr(),
                    html.P(
                        [
                            " by ",
                            blog_post["attributes"]["author"],
                            ", ",
                            blog_post["attributes"]["date"],
                        ]
                    ),
                    dcc.Markdown(blog_post["body"]),
                ]
            )

        self.layout = html.Div(
            header()
            + [
                dcc.Location(id="url", refresh=False),
                dbc.Container(
                    [
                        breadcrumb_layout([("Home", "/"), ("Blog", "")]),
                        dbc.Row(dbc.Col(body, lg=12)),
                    ]
                    + footer(),
                    style={"margin-bottom": "64px"},
                ),
            ]
        )


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
                    html.H1(blog_post["attributes"]["title"]),
                    href=f"/blog?post={blog_post['filename']}",
                    id=blog_post["filename"],
                ),
                html.Hr(),
                html.P(
                    [
                        " by ",
                        blog_post["attributes"]["author"],
                        ", ",
                        blog_post["attributes"]["date"],
                    ]
                ),
                dcc.Markdown(blog_post["body"]),
            ]


class BlogSection(MultiPageApp):
    def get_routes(self):
        return [
            Route(Blog, "Blog", "/"),
            Route(Post, "Post", "/post/"),
        ]
