### Steps to run the app

## clone your repo 
Clone your repository and navigate into it.

## Pull docker image
1-pull the docker image using the command: docker pull mahmoudmoumni/tf_mediapipe_jetson-nano:latest
I created this image based on the l4t-tensorflow:r32.7.1-tf2.7-py3 image tags for JetPack 4.6.1 (L4T R32.7.1) 
I build and installed mediapipe and bench of other libraries such as dlib and pyqt5 ,so you will have an image 
where you can develop other projects and use easily many libraries that are hard and tricky to install on host

## Edit the launch_container.sh script
First  make sure you camera is plugged in , and check the corresponding device number (/dev/video0 or/dev/video1 ...)
You can use command ls /dev/video* , then edit the file launch_container.sh and change your camera device number

## run the script launch_container.sh 
Use the command sudo /bin/bash launch_container.sh

Once the container shell is active, go to "/app/src"
using cd /app/src

## run the app
python3 app_main.py 
