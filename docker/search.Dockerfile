FROM bbs_base

USER root

# Install the app
ADD . /src
WORKDIR /src
RUN pip install .

# Set image version
LABEL maintainer="BBP-EPFL Machine Learning team <bbp-ou-machinelearning@groupes.epfl.ch>"
LABEL description="REST API Server for BBSearch"

# Add a user
RUN useradd --create-home serveruser
WORKDIR /home/serveruser
USER serveruser

# Download the NLTK libraries (for the current user)
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Run the entry point
# Note the "timeout" parameter. That's to let the server initialisation finish before
# gunicorn decides that the worker is not responsive and restarts it again.
# Might think about a better solution in the future... (initialize in a threading?)
EXPOSE 8080
ENTRYPOINT [\
"gunicorn", \
"--bind", "0.0.0.0:8080", \
"--workers", "1", \
"--timeout", "3600", \
"bbsearch.entrypoints:get_search_app()"]
