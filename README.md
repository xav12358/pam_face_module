# pam-facial-authentification

A pluggable authentication module which uses ML to detect and recognition face 
to autjentificate user on Unix. The database_generator generates features from
user face to setup the database. The setup.sh install the feature dataset of 
a given user. The pam module looks for matches between user in front of webcam 
and features in datasets associated to a username. 



Requirements
------------
- OpenCV 3.2+ including extra modules (opencv_contrib)
- PAM development packages (libpam0g and libpam0g-dev on ubuntu)
- dirent.h available in PATH (already installed on ubuntu)

Quickstart
----------
Clone this repository
```




