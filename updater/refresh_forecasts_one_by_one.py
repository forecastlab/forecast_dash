import argparse
import json
import os
import pickle
import shutil
import tempfile

from slugify import slugify

from download import download_data
from generate_search_details import generate_search_details
from generate_thumbnails import generate_static_thumbnail
from run_models import model_class_list, run_models


def write_json(path, data):
    with open(path, "w") as json_file:
        json.dump(data, json_file)


def replace_if_exists(source_path, destination_path):
    if not os.path.exists(source_path):
        raise FileNotFoundError(source_path)

    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    os.replace(source_path, destination_path)


def clear_nginx_cache(cache_path):
    if not os.path.isdir(cache_path):
        return

    for filename in os.listdir(cache_path):
        filepath = os.path.join(cache_path, filename)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)


def forecast_has_all_models(forecast_path):
    if not os.path.exists(forecast_path):
        return False

    with open(forecast_path, "rb") as forecast_file:
        forecast_data = pickle.load(forecast_file)

    expected_models = {model.name for model in model_class_list}
    actual_models = set(forecast_data.get("all_forecasts", {}).keys())
    return expected_models.issubset(actual_models)


def write_statistics(forecast_dir_path):
    statistics_path = os.path.join(forecast_dir_path, "statistics.pkl")
    statistics = {"models_used": [model.name for model in model_class_list]}

    with tempfile.NamedTemporaryFile(
        delete=False, dir=forecast_dir_path
    ) as temp_file:
        pickle.dump(statistics, temp_file)
        temp_statistics_path = temp_file.name

    replace_if_exists(temp_statistics_path, statistics_path)


def refresh_one_by_one(
    sources_path,
    download_dir_path,
    forecast_dir_path,
    thumbnail_dir_path,
    search_path,
    nginx_cache_path,
    skip_complete,
):
    with open(sources_path, "r") as sources_file:
        data_sources = json.load(sources_file)

    print("Downloading Data")
    download_data(sources_path, download_dir_path)

    os.makedirs(forecast_dir_path, exist_ok=True)
    os.makedirs(thumbnail_dir_path, exist_ok=True)

    for index, data_source in enumerate(data_sources, start=1):
        title = data_source["title"]
        filename = slugify(title)
        live_forecast_path = os.path.join(forecast_dir_path, f"{filename}.pkl")

        if skip_complete and forecast_has_all_models(live_forecast_path):
            print(
                f"[{index}/{len(data_sources)}] "
                f"Skipping complete forecast: {title}"
            )
            continue

        print(f"[{index}/{len(data_sources)}] Refreshing forecast: {title}")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_sources_path = os.path.join(temp_dir, "data_sources.json")

            write_json(temp_sources_path, [data_source])

            with tempfile.TemporaryDirectory(
                dir=forecast_dir_path, prefix=".refresh-"
            ) as temp_forecast_dir:
                run_models(
                    temp_sources_path,
                    download_dir_path,
                    temp_forecast_dir,
                )

                staged_forecast_path = os.path.join(
                    temp_forecast_dir, f"{filename}.pkl"
                )
                replace_if_exists(staged_forecast_path, live_forecast_path)

            with tempfile.TemporaryDirectory(
                dir=thumbnail_dir_path, prefix=".refresh-"
            ) as temp_thumbnail_dir:
                generate_static_thumbnail(
                    temp_sources_path, temp_thumbnail_dir
                )
                staged_thumbnail_path = os.path.join(
                    temp_thumbnail_dir, f"{filename}.pkl"
                )
                live_thumbnail_path = os.path.join(
                    thumbnail_dir_path, f"{filename}.pkl"
                )
                replace_if_exists(staged_thumbnail_path, live_thumbnail_path)

        print(f"[{index}/{len(data_sources)}] Finished forecast: {title}")

    write_statistics(forecast_dir_path)

    print("Updating search index")
    generate_search_details(sources_path, search_path)

    print("Clearing Cache")
    clear_nginx_cache(nginx_cache_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Refresh forecast pickle files one series at a time."
    )
    parser.add_argument(
        "--sources",
        default="../shared_config/data_sources.json",
        help="Path to the data sources JSON file.",
    )
    parser.add_argument(
        "--downloads",
        default="../data/downloads",
        help="Directory containing downloaded source data pickle files.",
    )
    parser.add_argument(
        "--forecasts",
        default="../data/forecasts",
        help="Directory containing live forecast pickle files.",
    )
    parser.add_argument(
        "--thumbnails",
        default="../data/thumbnails",
        help="Directory containing live thumbnail pickle files.",
    )
    parser.add_argument(
        "--search",
        default="../shared_config/search_a_series.json",
        help="Path to the search index JSON file.",
    )
    parser.add_argument(
        "--nginx-cache",
        default="/nginx_cache",
        help="Path to the nginx cache directory.",
    )
    parser.add_argument(
        "--skip-complete",
        action="store_true",
        help="Skip series that already contain every configured model.",
    )

    args = parser.parse_args()

    refresh_one_by_one(
        args.sources,
        args.downloads,
        args.forecasts,
        args.thumbnails,
        args.search,
        args.nginx_cache,
        args.skip_complete,
    )
