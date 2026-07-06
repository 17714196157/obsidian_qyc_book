## 常用功能示例
### django部署
```
gunicorn url.wsgi:application --bind 0.0.0.0:9191 --workers 4 --worker-class gevent --worker-connections 1000 
```
### django中间件
```python
# utils/log_middleware.py
class LogMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # 记录请求
        body = request.body.decode('utf-8', errors='ignore') if request.body else ''
        try:
            response = self.get_response(request)
            # 记录正常响应    
            ## 记录响应
            content = response.content.decode('utf-8', errors='ignore') if response.content else ''    
        except Exception as e:
            # 记录异常
            content = f"RES EXCEPTION {type(e).__name__}: {str(e)}"    

        finally:
            # 记录响应
            logger.debug(f"REQ {request.path}$$\n\n{body}\n\n$$RES$${response.status_code}$$\n\n{content}\n\n$$END")

            return response
```

```
# settings.py
MIDDLEWARE = [
    'middleware.log_middleware.LogMiddleware',  # 放最前面
    # ... 其他中间件
]
```
![[django中间件.png]]