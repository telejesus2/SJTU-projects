FROM ubuntu:xenial

RUN apt update && \
    apt install -y nodejs-legacy

COPY . /tmp
WORKDIR /tmp

CMD ["node", "backend_server.js"]
