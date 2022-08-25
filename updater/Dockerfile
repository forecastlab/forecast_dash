# -*- mode:sh -*-
# Dockerfile structure: put things at the top that won't change often.
#
# "Docker doesn't rebuild layers it's already built, and it caches the
# Dockerfile line by line. Try changing one character of a Dockerfile you've
# already built, and then rebuild. You'll notice that each step above that line
# will be recognized as already been built, but the line you changed (and each
# line following) will be rebuilt."
#   (from https://hackernoon.com/tips-to-reduce-docker-image-sizes-876095da3b34)
#
FROM python:3.9-bullseye

# Install cron and dependencies
RUN apt-get update && apt-get -y install cron

# Install and set timezone
RUN apt-get -y install tzdata &&\
    cp /usr/share/zoneinfo/Australia/Sydney /etc/localtime &&\
    echo "Australia/Sydney" > /etc/timezone

# Add crontab file in the cron directory
ADD updater-cron /etc/cron.d/updater-cron

# Give execution rights on the cron job
RUN chmod 0644 /etc/cron.d/updater-cron
# Apply cron job
RUN crontab /etc/cron.d/updater-cron

# Create the log file to be able to run tail
RUN touch /var/log/cron.log

# Add unstable as a source to get access to R binaries
RUN echo "deb http://deb.debian.org/debian testing main contrib non-free" > /etc/apt/sources.list && apt-get update

# --no-install-recommends reduces the install footprint substantially
ENV APTOPTS "-y -t testing --no-install-recommends"

# Install R and dependencies
COPY ./requirements-bin.txt .
RUN apt-get update &&  apt-get -y install build-essential libopenblas-base r-base r-cran-forecast

# Install remaining R packages from source
COPY ./requirements.R ./
RUN Rscript requirements.R

# Run the command on container startup
CMD ["cron", "-f"]
