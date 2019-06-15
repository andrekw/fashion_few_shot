FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04


RUN apt-get update && apt-get install -y python3 python3-pip adduser

ADD requirements.txt /tmp/
WORKDIR /tmp
RUN pip3 install -r requirements.txt

EXPOSE 8888
EXPOSE 6006

ENV PS1 'few_shot_docker \w $'
