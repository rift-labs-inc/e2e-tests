# E2E Tests 

## Dependencies

Foundry and Python3.10+ must be installed
```
python -m pip install -r circuits/requirements.txt
python -m pip install -r requirements.txt
```


## Run tests
```
python -m pytest circuits/tests
```

## Local Explorer 
docker run --rm -p 5100:80 --name otterscan -d --env ERIGON_URL="127.0.0.1:9000" otterscan/otterscan:latest
