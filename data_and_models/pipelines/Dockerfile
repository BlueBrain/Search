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

RUN conda install -c carta mysqlclient

# Revision can be a branch, sha, or tag
ARG BBS_REVISION=v0.0.2
ADD . /src
WORKDIR /src
RUN pip install git+file://$PWD@$BBS_REVISION

# Assume that we have Prodigy wheel there!
RUN pip install prodigy-1.10.4-cp36.cp37.cp38-cp36m.cp37m.cp38-linux_x86_64.whl