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

# Install R - must be present for rpy2
RUN apt-get -y install libopenblas-base r-base

COPY ./requirements.R ./
RUN Rscript requirements.R
RUN rm requirements.R

# Install Python Libraries
ENV PYTHONUNBUFFERED 1
COPY ./requirements.txt ./
RUN pip install -r requirements.txt
RUN rm requirements.txt

# Run the command on container startup
CMD ["cron", "-f"]