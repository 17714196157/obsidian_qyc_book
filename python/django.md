django部署
```
gunicorn url.wsgi:application --bind 0.0.0.0:9191 --workers 4 --worker-class gevent --worker-connections 1000 
```