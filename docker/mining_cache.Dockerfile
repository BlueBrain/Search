FROM bbs_base

USER root

# Install the app
ADD . /src
WORKDIR /src
RUN pip install -e .

# Set image version
LABEL maintainer="BBP-EPFL Machine Learning team <bbp-ou-machinelearning@groupes.epfl.ch>"
LABEL description="Creation of a Mining Cache for the Mining Server"


RUN chmod -R a+rwX /src

# Download the NLTK libraries (for the current user)
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Run the entry point
ENTRYPOINT ["/src/docker/mining_cache.sh"]
