#/usr/bin/env bash

# Start logging and cron services. Syslog messages will be in /var/log/syslog
service rsyslog start
service cron start

/usr/local/bin/docker-entrypoint.sh mysqld
