CURRENT_UID := $(shell id -u)
CURRENT_GID := $(shell id -g)

.PHONY = run

build-docker: Pipfile Pipfile.lock Dockerfile
	mkdir -p build
	cp Pipfile* build/
	cd build && sudo docker build -f ../Dockerfile -t 'kwandre/fashion_few_shot:latest' .

run: build/.DOCKERDONE
	sudo docker run --runtime nvidia -it --mount src=${CURDIR},target=/app,type=bind -p 8888:8888 -p 6006:6006 -u ${CURRENT_UID}:${CURRENT_GID} fashion_few_shot bash -c 'source /venv/bin/activate && cd /app && bash'
