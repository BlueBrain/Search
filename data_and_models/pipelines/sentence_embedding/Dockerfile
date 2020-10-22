FROM continuumio/miniconda3

ENV HTTP_PROXY='http://bbpproxy.epfl.ch:80/'
ENV HTTPS_PROXY='http://bbpproxy.epfl.ch:80/'
ENV http_proxy='http://bbpproxy.epfl.ch:80/'
ENV https_proxy='http://bbpproxy.epfl.ch:80/'

# Update conda, install additional system packages
RUN true \
	&& conda update conda \
	&& apt-get update \
	&& apt-get install -y gcc g++ build-essential vim libfontconfig1 wkhtmltopdf
RUN conda install -c carta mysqlclient

# Instal BlueBrainSearach -- revision can be a branch, sha, or tag
ARG BBS_REVISION=v0.0.4
ADD . /src
WORKDIR /src
RUN pip install git+file://$PWD@$BBS_REVISION


EXPOSE 8888

RUN groupadd -g 999 docker
RUN useradd --create-home --uid 1000 --gid docker bbsuser

WORKDIR /bbs
RUN rm -rf /bbs/tmp
ENTRYPOINT ["/bin/bash"]
