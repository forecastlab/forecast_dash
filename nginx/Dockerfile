FROM nginx

COPY ./nginx.conf /etc/nginx/conf.d/default.conf

# Set up certbot
COPY ./backports.list /etc/apt/sources.list.d/backports.list

RUN apt-get update

RUN apt-get -y install certbot python-certbot-nginx -t stretch-backports

RUN nginx

#RUN service nginx start

#RUN ls -all /var/log/nginx/

#RUN nginx

#RUN service nginx status -l
#
#RUN service nginx status

RUN certbot --nginx --non-interactive --agree-tos -m stephen.tierney@sydney.edu.au -d business-forecast-lab.com -d www.business-forecast-lab.com

RUN nginx -s stop

#
#RUN service nginx stop