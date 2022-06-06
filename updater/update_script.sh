#!/usr/bin/env bash

cd /updater
/usr/local/bin/pip install -r requirements.txt
/usr/local/bin/python -u update.py # devmode # Delete this to turn off dev mode. 