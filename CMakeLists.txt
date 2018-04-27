# Nous voulons un cmake "récent" pour utiliser les dernières fonctionnalités
cmake_minimum_required(VERSION 3.0)

project(pam_face_module)

file(GLOB SOURCES "src/*.cpp")
file(GLOB HEADERS "include/*.cpp")
include_directories( ${PROJECT_SOURCE_DIR}/include)

find_package(OpenCV 3.2.0 REQUIRED)

#SET("OpenCV_DIR" "/home/xavier/opencv/opencv-3.2.0/")

SET(OpenCV_LIBS  opencv_core opencv_highgui opencv_videoio opencv_imgproc )
add_executable(pam_face_module ${SOURCES} ${HEADERS})
target_link_libraries(pam_face_module
                       ${PROJECT_SOURCE_DIR}/lib/tensorflow/libtensorflow.so
                       ${PROJECT_SOURCE_DIR}/lib/tensorflow/libtensorflow_framework.so
                       ${OpenCV_LIBS})
