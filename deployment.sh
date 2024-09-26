#!/bin/sh

git reset --hard HEAD
git pull

if [ $(docker ps -f name=blue -q) ]
then
    ENV="green"
    OLD="blue"
else
    ENV="blue"
    OLD="green"
fi

echo $ENV > ./updater/container_version

echo "Starting "$ENV" container"
docker-compose --project-name=$ENV up -d

echo "Waiting..."
sleep 5s

echo "Stopping "$OLD" container"
docker-compose --project-name=$OLD stop
