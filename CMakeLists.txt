# Nous voulons un cmake "récent" pour utiliser les dernières fonctionnalités
cmake_minimum_required(VERSION 3.0)

project(pam_face_module)

#file(GLOB SOURCES "src/*.cpp")
#file(GLOB HEADERS "include/*.cpp")
#include_directories( ${PROJECT_SOURCE_DIR}/include)

#find_package(OpenCV 3.2.0 REQUIRED)

##SET("OpenCV_DIR" "/home/xavier/opencv/opencv-3.2.0/")

#SET(OpenCV_LIBS  opencv_core opencv_highgui opencv_videoio opencv_imgproc )
#add_executable(pam_face_module ${SOURCES} ${HEADERS})
#target_link_libraries(pam_face_module
#    ${PROJECT_SOURCE_DIR}/lib/tensorflow/libtensorflow.so
#    ${PROJECT_SOURCE_DIR}/lib/tensorflow/libtensorflow_framework.so
#    ${OpenCV_LIBS})


project("pam_face_module")

add_definitions("-std=c++11")

file(GLOB_RECURSE SOURCES "src/*.cpp" "src/pam_face_module/*.cpp" "test/*.cpp" "test/pam_face_module/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.h" "include/pam_face_module/*.h")
add_library("pam_face_module" ${SOURCES} ${HEADERS})
target_include_directories(
   "pam_face_module" PUBLIC "include" "test")

target_link_libraries("pam_face_module" ${OpenCV_LIBRARIES}
   "${CMAKE_CURRENT_LIST_DIR}/lib/tensorflow/libtensorflow.so"
   "${CMAKE_CURRENT_LIST_DIR}/lib/tensorflow/libtensorflow_framework.so")

####################################

file(GLOB SOURCES "test/pam_face_module/*.cpp" "test/*.cpp")
file(GLOB HEADERS "")
include_directories( ${PROJECT_SOURCE_DIR}/include)

find_package(OpenCV 3.2.0 REQUIRED)
find_package(GTest REQUIRED)
#SET("OpenCV_DIR" "/home/xavier/opencv/opencv-3.2.0/")

SET(OpenCV_LIBS  opencv_core opencv_highgui opencv_videoio opencv_imgproc opencv_imgcodecs )
add_executable("pam_face_module_test" ${SOURCES} ${HEADERS})
target_link_libraries("pam_face_module_test"
    "pam_face_module"
#    "/home/xavier/Bureau/developpement/NeuralNetwork/pam_face_module/lib/tensorflow/libtensorflow.so"
#    "/home/xavier/Bureau/developpement/NeuralNetwork/pam_face_module/lib/tensorflow/libtensorflow_framework.so"
#    ${PROJECT_SOURCE_DIR}/lib/tensorflow/libtensorflow.so
#    ${PROJECT_SOURCE_DIR}/lib/tensorflow/libtensorflow_framework.so
    ${OpenCV_LIBS}
    ${GTEST_LIBRARIES}
    pthread)
