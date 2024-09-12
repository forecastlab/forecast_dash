venv_name=venv

virtualenv -p python3.9 $venv_name
source $venv_name/bin/activate
pip install -r dash/requirements.txt
pip install -r updater/requirements.txt
pip install -r requirements_build.txt