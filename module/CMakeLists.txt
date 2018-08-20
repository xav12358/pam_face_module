# Nous voulons un cmake "récent" pour utiliser les dernières fonctionnalités
cmake_minimum_required(VERSION 3.0)

#project("database_generator")

#add_definitions("-std=c++11")

#file(GLOB SOURCES  "database_generator.cpp")
#file(GLOB HEADERS "")

#include_directories( "${PROJECT_SOURCE_DIR}/../library/include")

#add_executable("database_generator" ${SOURCES} ${HEADERS})
#target_link_libraries("database_generator"
#    "face_module")



project("pam_module")

add_definitions("-std=c++11")

file(GLOB SOURCES  "pam_module.cpp")
file(GLOB HEADERS "")


add_executable("pam_module" ${SOURCES} ${HEADERS})
target_link_libraries("pam_module"
    "face_module")

