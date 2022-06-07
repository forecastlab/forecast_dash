from flask import Flask
from multipage import Route, MultiPageApp
from pages import Index, Series, Search, Leaderboard
from pages_static import Methodology, About

from blog import BlogSection

home_route = (Index, "Business Forecast Lab", "/")

nav_routes = [
    (Search, "Find a Series", "/search/"),
    (Leaderboard, "Leaderboard", "/leaderboard/"),
    (BlogSection, "Blog", "/blog"),
    (Methodology, "Methodology", "/methodology/"),
    (About, "About", "/about/"),
]

dynamic_routes = [
    (Series, "Series", "/series/"),
]


class MyApp(MultiPageApp):
    def get_routes(self):

        return [
            Route(r[0], r[1], r[2])
            for r in [home_route] + nav_routes + dynamic_routes
        ]


server = Flask(__name__)

app = MyApp(name="", server=server, url_base_pathname="")

if __name__ == "__main__":
    server.run(host="0.0.0.0", port=8000, debug=True)
