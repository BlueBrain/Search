FROM mysql

# ENV HTTP_PROXY='http://bbpproxy.epfl.ch:80/'
# ENV HTTPS_PROXY='http://bbpproxy.epfl.ch:80/'
# ENV http_proxy='http://bbpproxy.epfl.ch:80/'
# ENV https_proxy='http://bbpproxy.epfl.ch:80/'

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.7 \
    python3-pip \
    vim

RUN pip3 install wheel setuptools
RUN pip3 install  sqlite3-to-mysql
