venv_name=venv

virtualenv --always-copy -p python3 $venv_name
source $venv_name/bin/activate
pip install -r dash/requirements.txt
pip install -r updater/requirements.txt
pip install -r requirements_build.txt