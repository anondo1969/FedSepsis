#!/bin/bash

# run with sudo

#author: Mahbub Ul Alam (mahbub@dsv.su.se)
#version: 1.0+
#copyright: Copyright (c) Mahbub Ul Alam (mahbub@dsv.su.se)
#license : MIT License

apt -y install python3-pip
pip3 install --upgrade torch==1.8.1
pip3 install --upgrade wandb
pip3 install pandas
pip3 install gpiozero
apt-get -y install python3-rpi.gpio
pip3 install sklearn



