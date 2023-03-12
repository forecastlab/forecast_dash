from download import download_data
from run_models import run_models
from generate_thumbnails import generate_static_thumbnail
import os
import shutil
import sys

if __name__ == "__main__":
    # # // This is helpful for testing
    # devmode = True if sys.argv[1] == "devmode" else False
    # data_sources = (
    #     "../shared_config/testing_data_sources.json"
    #     if devmode
    #     else "../shared_config/data_sources.json"
    # )

    # # This is helpful for testing //

    print("Downloading Data")
    download_data("../shared_config/data_sources.json", "../data/downloads")

    print("Running Models")
    run_models(
        "../shared_config/data_sources.json",
        "../data/downloads",
        "../data/forecasts",
    )

    print("Creating Thumbnails")
    generate_static_thumbnail(
        "../shared_config/data_sources.json", "../data/thumbnails"
    )

    if os.path.isdir("/nginx_cache"):
        print("Clearing Cache")
        for filename in os.listdir("/nginx_cache"):
            filepath = os.path.join("/nginx_cache", filename)
            try:
                shutil.rmtree(filepath)
            except OSError:
                os.remove(filepath)
