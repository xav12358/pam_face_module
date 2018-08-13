# Nous voulons un cmake "récent" pour utiliser les dernières fonctionnalités
cmake_minimum_required(VERSION 3.0)






#add_definitions("-std=c++11")

#file(GLOB_RECURSE SOURCES "library/src/face_module/*.cpp" )
#file(GLOB_RECURSE HEADERS "library/include/face_module/*.h")
#add_library("face_module" ${SOURCES} ${HEADERS})
#target_include_directories(
#   "face_module" PUBLIC "library/include" "test")



#find_package(OpenCV 3.2.0 REQUIRED)
#find_package(GTest REQUIRED)
#find_package(jsoncpp REQUIRED)

#set( JSONCPP_LIBRARIES    jsoncpp curl )
#SET(OpenCV_LIBS  opencv_core opencv_highgui opencv_videoio opencv_imgproc opencv_imgcodecs )

#target_link_libraries("face_module"
#    ${OpenCV_LIBRARIES}
#   "${CMAKE_CURRENT_LIST_DIR}/library/lib/tensorflow/libtensorflow.so"
#   "${CMAKE_CURRENT_LIST_DIR}/library/lib/tensorflow/libtensorflow_framework.so"
#   ${OpenCV_LIBS}
#   ${GTEST_LIBRARIES}
#   ${JSONCPP_LIBRARIES}
#   pthread)

#####################################

project("pam_face_module")

add_subdirectory(library)
add_subdirectory(test)
add_subdirectory(module)

#add_definitions("-std=c++11")

#file(GLOB SOURCES "test/face_module/*.cpp" "test/*.cpp")
#file(GLOB HEADERS "")
#include_directories( "${PROJECT_SOURCE_DIR}/library/include")

#add_executable("pam_face_module_test" ${SOURCES} ${HEADERS})
#target_link_libraries("pam_face_module_test"
#    "face_module")
