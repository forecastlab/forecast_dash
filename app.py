from flask import Flask
from multipage import Route, MultiPageApp
from pages import Index, Series, Filter, Methodology, About


class MyApp(MultiPageApp):
    def get_routes(self):

        return [
            Route(Index, "index", "/"),
            Route(Filter, "filter", "/filter/"),
            Route(Series, "series", "/series/"),
            Route(Methodology, "methodology", "/methodology/"),
            Route(About, "about", "/about/"),
        ]


server = Flask(__name__)

app = MyApp(name="", server=server, url_base_pathname="")

if __name__ == "__main__":
    server.run(host="127.0.0.1")
