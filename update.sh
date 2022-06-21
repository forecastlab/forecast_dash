#!/usr/bin/env bash

git reset --hard HEAD
git pull
touch shared_config/acme.json & chmod 600 shared_config/acme.json
docker-compose down -f docker-compose.traefik.yml 
docker-compose down
docker-compose up -f docker-compose.traefik.yml --force-recreate --build -d
# docker-compose up --force-recreate --build -d
chmod 600 ./deployment.sh
. ./deployment.sh # run in global?
webhook -hooks hooks.json & # ensure installed: sudo apt-get install webhook
