FROM mysql:8

ENV HTTP_PROXY="http://bbpproxy.epfl.ch:80"
ENV HTTPS_PROXY="http://bbpproxy.epfl.ch:80"
ENV http_proxy="http://bbpproxy.epfl.ch:80"
ENV https_proxy="http://bbpproxy.epfl.ch:80"

ENV TZ="Europe/Zurich"
RUN apt-get update && apt-get install -y --no-install-recommends man rsyslog cron vim less procps

# Disable kernel logging because it doesn't work here, see https://stackoverflow.com/a/60265997/2804645
RUN sed -i '/imklog/s/^/#/' /etc/rsyslog.conf

# Limit incremental binary log to 7 days. This is a system variable and has to
# go in the [mysqld] section, which is in docker.cnf
# Accordingly it would make sense to do file dumps every 7 days
RUN echo "binlog_expire_logs_seconds = 604800" >> /etc/mysql/conf.d/docker.cnf

COPY docker/mysql-files/start.sh /root/start.sh
COPY docker/mysql-files/make-backup /usr/local/bin/make-backup
RUN \
chmod +x /usr/local/bin/make-backup &&\
chmod +x /root/start.sh
VOLUME ["/backup"]
CMD ["/root/start.sh"]

