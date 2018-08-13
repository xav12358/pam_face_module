# Nous voulons un cmake "récent" pour utiliser les dernières fonctionnalités
cmake_minimum_required(VERSION 3.0)

project("pam_face_module_test")

add_definitions("-std=c++11")

file(GLOB SOURCES "face_module/*.cpp" "*.cpp")
file(GLOB HEADERS "")
include_directories( "${PROJECT_SOURCE_DIR}/../library/include")

add_executable("pam_face_module_test" ${SOURCES} ${HEADERS})
target_link_libraries("pam_face_module_test"
    "face_module")
