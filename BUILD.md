# Creating a Local Development Environment

For most local development it is not necessary to set up docker. 
Use the following script to create a python virtual environment and install
the required dependencies 

    ./create_local_env.sh

This only needs to be done once. Make sure you are working in that virtual
environment.

    source venv/bin/activate

# Run the dash app
From the dash/ directory
    python app.py
and direct your browser to http://0.0.0.0:5000/ .

# Run the updater
From the updater/ directory
    python download.py
to fetch new data.
    R -f requirements.R
to install new R packages and
    python update.py
to run the models.

# Build and run docker images
Takes >3GB disk and 6m15s on my machine:

    rm -rf /var/lib/docker
    systemctl restart docker
    docker network create web
    docker-compose down
    docker-compose up -d

Use
    docker container ls
to see running containers by name, and
    docker logs <NAME>
to see the output of a running docker instance.


# Local build

Before pushing commits you should locally build to run tests, check formatting
and documentation. Use the following command to run the build

    ./pre_push.sh

# Continuous Integration - Travis

CI is performed on Travis. The configuration file is

    .travis.yml
    
A slight modification needed to be made to the normal Travis config in order
for Python 3.7 to work
https://github.com/travis-ci/travis-ci/issues/9069#issuecomment-425720905

# PEP8 Formatting

To check PEP8 formatting we use flake8. Run flake8 by using the following
command, which will automatically discover the configuration in ``setup.cfg``

    flake8
    
To automatically format our code to PEP8 standard we use black. Run black by
using the following command which will automatically discover the configuration
in ``pyproject.toml``

    black ./    

black will not reformat comments, so it is important that you run flake8
locally to discover any issues before pushing.