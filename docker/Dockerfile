FROM continuumio/miniconda3

# ENV HTTP_PROXY='http://bbpproxy.epfl.ch:80/'
# ENV HTTPS_PROXY='http://bbpproxy.epfl.ch:80/'
# ENV http_proxy='http://bbpproxy.epfl.ch:80/'
# ENV https_proxy='http://bbpproxy.epfl.ch:80/'

RUN \
conda update conda && \
apt-get update && \
apt-get install -y gcc g++

# Install sent2vec
RUN pip install Cython numpy
RUN pip install git+https://github.com/epfml/sent2vec

# Install requirements.txt
RUN mkdir -p /bbs/tmp
COPY /requirements.txt /bbs/tmp
WORKDIR /bbs/tmp
RUN pip install -r requirements.txt

# Install PyTorch
# RUN conda install pytorch torchvision -c pytorch

# HTML to PDF
RUN wget https://github.com/wkhtmltopdf/wkhtmltopdf/releases/download/0.12.3/wkhtmltox-0.12.3_linux-generic-amd64.tar.xz
RUN tar vxf wkhtmltox-0.12.3_linux-generic-amd64.tar.xz
RUN cp wkhtmltox/bin/* /usr/local/bin/

# jupyterlab-manager
# RUN conda remove wrapt
RUN conda install -c conda-forge nodejs
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager

WORKDIR /bbs
RUN rm -rf /bbs/tmp
CMD /bin/bash

