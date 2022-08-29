#!/usr/bin/env bash

cd /updater
/usr/local/bin/pip install -r requirements.txt
/usr/local/bin/python -u update.py
/usr/local/bin/python -u generate_search_details.py 
