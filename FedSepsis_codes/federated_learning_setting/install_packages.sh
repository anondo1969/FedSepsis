#!/bin/bash

# run with sudo

apt -y install python3-pip
pip3 install --upgrade torch==1.8.1
pip3 install --upgrade wandb
pip3 install pandas
pip3 install gpiozero
apt-get -y install python3-rpi.gpio
pip3 install sklearn



