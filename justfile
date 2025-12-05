build-dev-image:
    docker build -t mpmi tools -f tools/Dockerfile.dev

run-dev-container: build-dev-image
    docker run --rm -it -v$PWD:/home/devel/mpmi -v/tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY mpmi:latest

build-shaders:
    glslc src/display_engine/shaders/shader.vert -o data/shaders/vert.spv
    glslc src/display_engine/shaders/shader.frag -o data/shaders/frag.spv