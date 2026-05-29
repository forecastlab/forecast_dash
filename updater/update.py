from download import download_data
from run_models import run_models
from generate_thumbnails import generate_static_thumbnail
from generate_search_details import generate_search_details
import json
import os
import shutil
import sys
from slugify import slugify


def cleanup_orphaned_pickles(sources_path, directories):
    with open(sources_path) as data_sources_json_file:
        data_sources = json.load(data_sources_json_file)

    expected_files = {
        f"{slugify(data_source['title'])}.pkl" for data_source in data_sources
    }
    expected_files.add("statistics.pkl")

    for directory in directories:
        if not os.path.isdir(directory):
            continue

        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if not os.path.isfile(filepath):
                continue
            if not filename.endswith(".pkl"):
                continue
            if filename in expected_files:
                continue

            print(f"Removing orphaned file: {filepath}")
            os.remove(filepath)

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

    print("Cleaning orphaned forecast and thumbnail files")
    cleanup_orphaned_pickles(
        "../shared_config/data_sources.json",
        ["../data/forecasts", "../data/thumbnails"],
    )

    print("Updating search index")
    generate_search_details(
        "../shared_config/data_sources.json",
        "../shared_config/search_a_series.json",
    )

    if os.path.isdir("/nginx_cache"):
        print("Clearing Cache")
        for filename in os.listdir("/nginx_cache"):
            filepath = os.path.join("/nginx_cache", filename)
            try:
                shutil.rmtree(filepath)
            except OSError:
                os.remove(filepath)
