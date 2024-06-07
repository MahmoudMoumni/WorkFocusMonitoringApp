# Introduction
Computer vision based pyqt app which aims at measuring the percentage of user's focus on work and calculates the distraction time either by using mobile phone or not looking at work environment.

I used the yolov4-tiny tensorflow implementation inspired from this repo :[a link](https://github.com/hunglc007/tensorflow-yolov4-tflite)

I basically check if the user is holding his mobile phone using yolov4-tiny model and check if the user is looking at work environment by both eyes detection using mediapipe.


## Demo
[![Watch the video](https://img.youtube.com/vi/ZHPmhk4Sm3Q/0.jpg)](https://www.youtube.com/watch?v=ZHPmhk4Sm3Q)



## Environment

Jetson nano 4GB
jetpack 4.6.4

## Teck Stack
 - Tensorflow
 - Mediapipe
 - Yolov4-tiny
 - PyQT5
 - Docker 

# Steps to run the app
## clone your repo 
Clone your repository and navigate into it.

## Pull docker image
1-pull the docker image using the command: 
```bash
docker pull mahmoudmoumni/tf_mediapipe_jetson-nano:latest
```
I created this image based on the l4t-tensorflow:r32.7.1-tf2.7-py3 image tags for JetPack 4.6.1 (L4T R32.7.1) 
I build and installed mediapipe and bench of other libraries such as dlib and pyqt5 ,so you will have an image 
where you can develop other projects and use easily many libraries that are hard and tricky to install on host

## Edit the launch_container.sh script
First  make sure you camera is plugged in , and check the corresponding device number (/dev/video0 or/dev/video1 ...)
You can use command
```bash
ls /dev/video* 
```
then edit the file launch_container.sh and change your camera device number

## run the script launch_container.sh 
Use the command 
```bash
sudo /bin/bash launch_container.sh
```
Once you are on  container's terminall, go to "/app/src" using 
```bash
cd /app/src
```
## run the app
run the app_main.py file to launch the app and then wait for few seconds to load yolo and mediapipe models
```bash
python3 app_main.py 
```


