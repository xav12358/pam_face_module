#!/bin/sh


#### build project

mkdir ./build
cd ./build
cmake ..
make 


#### then create the file and copy module to the directory
sudo mkdir -p /etc/pam-face_module
sudo cp ./module/libpam_face_module.so /lib/x86_64-linux-gnu/security/pam_face_module.so
