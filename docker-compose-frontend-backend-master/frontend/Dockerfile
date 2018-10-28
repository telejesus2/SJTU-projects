FROM ubuntu:xenial

RUN apt update && \
    apt install -y nodejs-legacy

COPY . /tmp
WORKDIR /tmp

EXPOSE 8888

CMD ["node", "frontend_server.js"]