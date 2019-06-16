FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

RUN apt-get update && apt-get install -y python3 python3-pip

ENV LANG C.UTF-8
ENV HOME /tmp

COPY requirements.txt /tmp/
WORKDIR /tmp
RUN pip3 install -r requirements.txt

RUN echo export PS1="\"\[\e[0;37m\]few_shot: \[\e[0;31m\]\w\[\e[0;31m\] > \[\e[0m\]\"" > /etc/bash.bashrc && chmod a+rwx /etc/bash.bashrc

EXPOSE 8888
EXPOSE 6006
