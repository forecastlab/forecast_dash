FROM python:3

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

# Install R - must be present for rpy2
RUN apt-get -y -t testing install gcc-8-base libopenblas-base r-base

# Install binary R packages
COPY ./requirements-bin.txt .
RUN cat requirements-bin.txt | xargs apt-get -y -t testing install

# Install remaining R packages from source
COPY ./requirements.R ./
RUN Rscript requirements.R

# Run the command on container startup
CMD ["cron", "-f"]