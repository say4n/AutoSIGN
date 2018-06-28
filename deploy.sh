mkdir .log 2> /dev/null
DEBUG=0 authbind gunicorn -b 0.0.0.0:80 main:app --access-logfile .log/access.log --error-logfile .log/general.log
