FROM continuumio/miniconda3

# ENV HTTP_PROXY='http://bbpproxy.epfl.ch:80/'
# ENV HTTPS_PROXY='http://bbpproxy.epfl.ch:80/'
# ENV http_proxy='http://bbpproxy.epfl.ch:80/'
# ENV https_proxy='http://bbpproxy.epfl.ch:80/'

# Update conda, install additional system packages
RUN true \
	&& conda update conda \
	&& apt-get update \
	&& apt-get install -y gcc g++ build-essential vim

# HTML to PDF
RUN wget https://github.com/wkhtmltopdf/wkhtmltopdf/releases/download/0.12.3/wkhtmltox-0.12.3_linux-generic-amd64.tar.xz
RUN tar vxf wkhtmltox-0.12.3_linux-generic-amd64.tar.xz
RUN cp wkhtmltox/bin/* /usr/local/bin/

# Install sent2vec
RUN pip install Cython numpy
RUN pip install git+https://github.com/epfml/sent2vec

# Install requirements.txt
RUN mkdir -p /bbs/tmp
COPY /requirements.txt /bbs/tmp
RUN pip install -r /bbs/tmp/requirements.txt

# Install PyTorch
# RUN conda install pytorch torchvision -c pytorch

# Jupyter
# RUN conda remove wrapt
RUN conda install -c conda-forge nodejs
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager
RUN jupyter labextension install @jupyterlab/toc
EXPOSE 8888

# SciBERT
RUN true \
	&& git clone https://github.com/allenai/scibert /scibert \
	&& pip install git+http://github.com/ibeltagy/allennlp@fp16_and_others#egg=allennlp \
	&& pip install jsonlines lxml
ENV PYTHONPATH=$PYTHONPATH:/scibert

RUN groupadd -g 999 docker
RUN useradd --create-home --uid 211679 --gid docker sschmidt

WORKDIR /bbs
RUN rm -rf /bbs/tmp
ENTRYPOINT exec /bin/bash

