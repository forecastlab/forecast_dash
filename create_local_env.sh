venv_name=venv

virtualenv -p python3 $venv_name
source $venv_name/bin/activate
pip install -r app/requirements.txt