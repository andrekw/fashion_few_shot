IMAGE_NAME := fashion_few_shot  #'kwandre/fashion_few_shot:latest'

CURRENT_UID := $(shell id -u)
CURRENT_GID := $(shell id -g)

.PHONY = run run-notebook create_user

build-docker: Pipfile Pipfile.lock Dockerfile
	mkdir -p build
	pipenv run pip freeze > build/requirements.txt
	cd build && sudo docker build -f ../Dockerfile -t ${IMAGE_NAME} .

run:
	sudo docker run --runtime nvidia -it --mount src=${CURDIR},target=/app,type=bind -u ${CURRENT_UID}:${CURRENT_GID} ${IMAGE_NAME} bash -c 'cd /app && bash'

run-notebook:
	sudo docker run --runtime nvidia -it --mount src=${CURDIR},target=/app,type=bind -p 8888:8888 -u ${CURRENT_UID}:${CURRENT_GID} ${IMAGE_NAME} bash -c 'cd /app && PYTHONPATH=. HOME=/tmp jupyter notebook --notebook-dir=./notebooks/ --ip=0.0.0.0'
