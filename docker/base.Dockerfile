FROM python:3.6

# ENV HTTP_PROXY='http://bbpproxy.epfl.ch:80/'
# ENV HTTPS_PROXY='http://bbpproxy.epfl.ch:80/'
# ENV http_proxy='http://bbpproxy.epfl.ch:80/'
# ENV https_proxy='http://bbpproxy.epfl.ch:80/'

# Install system packages
RUN apt-get update && apt-get install -y \
    git \
    gcc g++ \
    build-essential \
    vim htop \
    nodejs npm \
    libfontconfig1 wkhtmltopdf \
    default-libmysqlclient-dev

# Upgrade pip
RUN pip install --upgrade pip

# Install Jupyter & IPython
RUN true \
    && pip install ipython jupyter jupyterlab ipywidgets \
    && jupyter nbextension enable --py widgetsnbextension \
    && jupyter labextension install @jupyter-widgets/jupyterlab-manager \
    && jupyter labextension install @jupyterlab/toc
EXPOSE 8888

# Install BBS requirements
COPY requirements.txt /tmp
RUN true \
    && pip install Cython numpy \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt \
    && jupyter-lab build --name="BBS | Base"

# Download the scispaCy models
RUN pip install \
    https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_ner_craft_md-0.2.5.tar.gz \
    https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_ner_jnlpba_md-0.2.5.tar.gz \
    https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_ner_bc5cdr_md-0.2.5.tar.gz \
    https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_ner_bionlp13cg_md-0.2.5.tar.gz \
    https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_core_sci_lg-0.2.5.tar.gz

# Configure Jupyter (for the root)
# Set --allow-root --no-browser --ip=0.0.0.0
RUN true \
  && jupyter-lab --generate-config \
  && sed -i"" \
     -e "s/#c.NotebookApp.ip = 'localhost'/c.NotebookApp.ip = '0.0.0.0'/g" \
     -e "s/#c.NotebookApp.open_browser = True/c.NotebookApp.open_browser = False/g" \
     -e "s/#c.NotebookApp.allow_root = False/c.NotebookApp.allow_root = True/g" \
     /root/.jupyter/jupyter_notebook_config.py

# Download the NLTK data (for the root)
RUN python -m nltk.downloader punkt stopwords

# Add and select a non-root user (bbsuser)
RUN groupadd -g 999 docker
RUN useradd --create-home --uid 1000 --gid docker bbsuser
USER bbsuser

# Configure Jupyter (for the bbsuser) 
# Set --no-browser --ip=0.0.0.0
RUN true \
  && jupyter-lab --generate-config \
  && sed -i"" \
     -e "s/#c.NotebookApp.ip = 'localhost'/c.NotebookApp.ip = '0.0.0.0'/g" \
     -e "s/#c.NotebookApp.open_browser = True/c.NotebookApp.open_browser = False/g" \
     /home/bbsuser/.jupyter/jupyter_notebook_config.py

# Download the NLTK data (for the bbsuser)
RUN python -m nltk.downloader punkt stopwords

WORKDIR /home/bbsuser
ENTRYPOINT ["bash"]

