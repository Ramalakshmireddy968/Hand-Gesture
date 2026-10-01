This project is used to control the robotic hand manipulator of 5 DOF  by through hand gestures 

the tools we use in this project are opencv , mediapipe of both tasks version and solution version 

gest_to_signal.py , file reads the gestures from the both hands of the user, based on the gesture this file creates signal for all 6 servo motors and send the signal to the esp32 through usb port

forward_kinematics.py it is used to calculate the position of the end effector , through forward kinematics

