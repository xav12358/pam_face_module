# Nous voulons un cmake "récent" pour utiliser les dernières fonctionnalités
cmake_minimum_required(VERSION 3.0)

project("pam_face_module")

add_subdirectory(library)
add_subdirectory(test)
add_subdirectory(module)

