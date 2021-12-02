from download import download_data
from run_models import run_models
import os
import shutil

print("Downloading Data")
download_data("../shared_config/data_sources.json", "../data/downloads")

print("Running Models")
run_models(
    "../shared_config/data_sources.json",
    "../data/downloads",
    "../data/forecasts",
)

print("Clearing Cache")
for filename in os.listdir("/nginx_cache"):
    filepath = os.path.join("/nginx_cache", filename)
    try:
        shutil.rmtree(filepath)
    except OSError:
        os.remove(filepath)
