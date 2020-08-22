FROM bbs_base

# Install the app
ADD . /src
WORKDIR /src
RUN pip install . gunicorn

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
EXPOSE 8080
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:8080", "bbsearch.entrypoints:run_search_server()"]
