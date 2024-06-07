#!/bin/sh

# Get the current absolute folder path
installation_path=$(pwd)

# Allow Docker to access the X server
xhost +local:docker

# Run the Docker container with the specified parameters
docker run -it --rm --device=/dev/video0 --privileged -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix --runtime nvidia --network host \
    -v "$installation_path:/app/src" tf_mediapipe_jetson-nano
