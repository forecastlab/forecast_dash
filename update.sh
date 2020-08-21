#!/usr/bin/env bash

git reset --hard HEAD
git pull
touch shared_config/acme.json & chmod 600 shared_config/acme.json
docker-compose down
docker-compose up --force-recreate --build -d