FROM continuumio/miniconda3

# ENV HTTP_PROXY='http://bbpproxy.epfl.ch:80/'
# ENV HTTPS_PROXY='http://bbpproxy.epfl.ch:80/'
# ENV http_proxy='http://bbpproxy.epfl.ch:80/'
# ENV https_proxy='http://bbpproxy.epfl.ch:80/'

# Update conda, install additional system packages
RUN true \
	&& conda update conda \
	&& apt-get update \
	&& apt-get install -y gcc g++ build-essential vim libfontconfig1 wkhtmltopdf

# Install requirements.txt
RUN mkdir -p /bbs/tmp
COPY /requirements.txt /bbs/tmp
RUN pip install Cython numpy
RUN pip install -r /bbs/tmp/requirements.txt

# Jupyter
RUN conda install -c conda-forge nodejs
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager
RUN jupyter labextension install @jupyterlab/toc
EXPOSE 8888

RUN groupadd -g 999 docker
RUN useradd --create-home --uid 1000 --gid docker bbsuser

WORKDIR /bbs
RUN rm -rf /bbs/tmp
ENTRYPOINT exec /bin/bash
