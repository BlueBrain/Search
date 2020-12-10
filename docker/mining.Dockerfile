FROM bbs_base

USER root

# Install the app
ADD . /src
WORKDIR /src
RUN pip install .

# Set image version
LABEL maintainer="BBP-EPFL Machine Learning team <bbp-ou-machinelearning@groupes.epfl.ch>"
LABEL description="REST API Server for Test Mining"


RUN chmod 777 -R /src/

## Add a user
#RUN useradd --create-home serveruser
#WORKDIR /home/serveruser
#USER serveruser

# Download the NLTK libraries (for the current user)
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Run the entry point
EXPOSE 8080
ENTRYPOINT ["/src/docker/mining.sh"]
