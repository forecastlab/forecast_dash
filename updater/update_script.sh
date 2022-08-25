#!/usr/bin/env bash

cd /updater
/usr/local/bin/pip install -r requirements.txt
/usr/local/bin/python -u update.py > update_output.log