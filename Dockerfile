FROM ubuntu:latest
LABEL authors="hivaze"

ENTRYPOINT ["top", "-b"]