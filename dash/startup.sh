#!/usr/bin/env bash

cd /dash
/usr/local/bin/pip install -r requirements.txt
gunicorn -b 0.0.0.0:80 app:server
