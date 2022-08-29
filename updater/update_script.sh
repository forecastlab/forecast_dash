#!/usr/bin/env bash

cd /updater
/usr/local/bin/pip install -r requirements.txt
/usr/bin/r requirements.R
/usr/local/bin/python -u update.py > updater_ouput.log
