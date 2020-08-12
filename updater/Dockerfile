FROM python:3

# Install cron and dependencies
RUN apt-get update && apt-get -y install cron

# Install and set timezone
RUN apt-get -y install tzdata &&\
    cp /usr/share/zoneinfo/Australia/Sydney /etc/localtime &&\
    echo "Australia/Sydney" > /etc/timezone

# Add crontab file in the cron directory
ADD crontab /etc/cron.d/simple-cron
WORKDIR /src
COPY . /src

# Give execution rights on the cron job
RUN chmod 0644 /etc/cron.d/simple-cron
RUN chmod 777 /usr/local/lib/python3.7/site-packages

# Create the log file to be able to run tail
RUN touch /var/log/cron.log

# Add unstable as a source to get access to R binaries
RUN echo "deb http://deb.debian.org/debian unstable main contrib non-free" > /etc/apt/sources.list && apt-get update

# Install R - must be present for rpy2
RUN apt-get -y -t unstable install gcc-8-base libopenblas-base r-base

# Install binary R packages
COPY ./requirements-bin.txt .
RUN cat requirements-bin.txt | xargs apt-get install -y -qq

# Install remaining R packages from source
COPY ./requirements.R ./
RUN Rscript requirements.R

# Install Python Libraries
ENV PYTHONUNBUFFERED 1
COPY ./requirements.txt ./
RUN pip install -r requirements.txt
RUN rm requirements.txt

# Run the command on container startup
CMD ["cron", "-f"]