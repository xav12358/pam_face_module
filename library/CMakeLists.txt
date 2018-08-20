# Nous voulons un cmake "récent" pour utiliser les dernières fonctionnalités
cmake_minimum_required(VERSION 3.0)

project("face_module")

add_definitions("-std=c++11" "-Wno-write-strings")

file(GLOB_RECURSE SOURCES "*.cpp"")
file(GLOB_RECURSE HEADERS "*.h"")
add_library("face_module" SHARED ${SOURCES} ${HEADERS})
target_include_directories(
    "face_module" PUBLIC "include" )

find_package(OpenCV 3.2.0 REQUIRED)
find_package(jsoncpp REQUIRED)

set( JSONCPP_LIBRARIES    jsoncpp curl )
SET(OpenCV_LIBS  opencv_core opencv_highgui opencv_videoio opencv_imgproc opencv_imgcodecs )

target_link_libraries("face_module"
    "${CMAKE_CURRENT_LIST_DIR}/lib/tensorflow/libtensorflow.so"
    "${CMAKE_CURRENT_LIST_DIR}/lib/tensorflow/libtensorflow_framework.so"
    ${OpenCV_LIBS}
    ${JSONCPP_LIBRARIES}
    pthread)
