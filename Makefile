

.PHONY = run

build/.DOCKERDONE: Pipfile Pipfile.lock Dockerfile
	mkdir -p build
	cp Pipfile* build/
	cd build && sudo docker build -f ../Dockerfile -t fashion_few_shot .
	touch build/.DOCKERDONE

run: build/.DOCKERDONE
	sudo docker run --runtime nvidia -it --mount src=${CURDIR},target=/app,type=bind -p 8888:8888 -p 6006:6006 -u $(id -u):$(id -g) fashion_few_shot bash
