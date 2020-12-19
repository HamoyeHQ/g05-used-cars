#! /usr/bin/bash

docker build -t apache-process-image .
docker run -it apache-process-image