Production:
```
uvicorn main:api --host 0.0.0.0 --port 8008 --log-level info --root-path /api --workers 32
```

Debug locally:
```
uvicorn main:api --port 8008
```
