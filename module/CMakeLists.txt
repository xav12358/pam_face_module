# Nous voulons un cmake "récent" pour utiliser les dernières fonctionnalités
cmake_minimum_required(VERSION 3.0)

project("database_generator")

add_definitions("-std=c++11" )

file(GLOB SOURCES  "database_generator.cpp")
file(GLOB HEADERS "")
find_package(OpenCV 3.2.0 REQUIRED)

SET(OpenCV_LIBS  opencv_core opencv_highgui opencv_videoio opencv_imgproc opencv_imgcodecs )

add_executable("database_generator" ${SOURCES} ${HEADERS})
target_link_libraries("database_generator"
    ${OpenCV_LIBS}
 "face_module")




project("pam_module")

add_definitions("-std=c++11")

file(GLOB SOURCES  "pam_module.cpp")
file(GLOB HEADERS "")

add_library("pam_face_module" SHARED ${SOURCES} ${HEADERS})
target_link_libraries("pam_face_module"
    -lpam
    -lpam_misc
    "face_module")

