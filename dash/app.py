from flask import Flask
from multipage import Route, MultiPageApp
from pages import Index, Series, Search, Leaderboard
from blog import BlogSection
from pages_static import Methodology, About


class MyApp(MultiPageApp):
    def get_routes(self):

        return [
            Route(Index, "index", "/"),
            Route(BlogSection, "blog", "/blog"),
            Route(Series, "series", "/series/"),
            Route(Search, "search", "/search/"),
            Route(Leaderboard, "stats", "/leaderboard/"),
            Route(Methodology, "methodology", "/methodology/"),
            Route(About, "about", "/about/"),
        ]


server = Flask(__name__)

app = MyApp(name="", server=server, url_base_pathname="")

if __name__ == "__main__":
    server.run(host="0.0.0.0", debug=True)
