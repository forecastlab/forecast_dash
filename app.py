from flask import Flask
from multipage import Route, MultiPageApp
from pages import Index, Series

class MyApp(MultiPageApp):
    def get_routes(self):

        return [
            Route(Index, "index", "/"),
            Route(Series, "series", "/series"),
        ]

server = Flask(__name__)

app = MyApp(name="", server=server, url_base_pathname="")

if __name__ == "__main__":
    server.run(host="0.0.0.0")