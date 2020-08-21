from download import download_data
from run_models import run_models

print("Downloading Data")
download_data("../shared_config/data_sources.json", "../data/downloads")

print("Running Models")
run_models(
    "../shared_config/data_sources.json",
    "../data/downloads",
    "../data/forecasts",
)
