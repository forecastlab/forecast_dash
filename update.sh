#!/usr/bin/env bash

git reset --hard HEAD
git pull
chmod 600 traefik/acme.json
docker-compose down
docker-compose up -d