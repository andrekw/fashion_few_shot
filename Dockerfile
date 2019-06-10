FROM tensorflow/tensorflow:nightly-gpu-py3

RUN apt-get -y install python3-venv

RUN python3 -m venv /venv

RUN /venv/bin/pip3 install pipenv

ADD Pipfile* /tmp/

WORKDIR /tmp
RUN source /venv/bin/activate && PIPENV_VERBOSITY=-1 pipenv install --dev
