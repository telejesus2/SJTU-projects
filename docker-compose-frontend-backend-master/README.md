# Frontend and Backend Web

## Build Images
- `docker image build -t backend backend`
- `docker image build -t frontend frontend`


## Docker Deployment
### Create Network
- `docker network create frontbackend`

### Launch Containers
- `docker container run --name backend -d --net=frontbackend backend`
- `docker container run --name frontend -d --net=frontbackend -p 8888:8888 frontend`


## docker-compose Deployment
