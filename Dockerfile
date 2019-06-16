FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

RUN apt-get update && apt-get install -y python3 python3-pip && pip3 install pipenv

ENV LANG C.UTF-8
ENV HOME /tmp

COPY requirements.txt /tmp/
WORKDIR /tmp
RUN pip3 install -r requirements.txt

RUN echo "PS1=\"few_shot_docker: \w >\"" > /etc/bash.bashrc && chmod a+rwx /etc/bash.bashrc

EXPOSE 8888
EXPOSE 6006
