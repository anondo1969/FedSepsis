#!/usr/bin/env python
# coding: utf-8
'''
@author: Mahbub Ul Alam (mahbub@dsv.su.se)
@version: 1.0+
@copyright: Copyright (c) Mahbub Ul Alam (mahbub@dsv.su.se)
@license : MIT License
'''
import os
from gpiozero import CPUTemperature
import torch

def getFreeDescription():
    free = os.popen("free -h")
    i = 0
    while True:
        i = i + 1
        line = free.readline()
        if i == 1:
            return line.split()[0:7]


def getFree():
    free = os.popen("free -h")
    i = 0
    while True:
        i = i + 1
        line = free.readline()
        if i == 2:
            return line.split()[0:7]


def printPerformance():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    temperature = 0
    if device == 'cpu':
        cpu = CPUTemperature()
        temperature = cpu.temperature
    else:
        temperature = float(os.popen("cat /sys/devices/virtual/thermal/thermal_zone0/temp").read().strip('\n'))/1000
        

    print()
    print('Current system status:')
    print()
    print("temperature: " + str(temperature))

    description = getFreeDescription()
    mem = getFree()

    print(description[0] + " : " + mem[1])
    print(description[1] + " : " + mem[2])
    print(description[2] + " : " + mem[3])
    print(description[3] + " : " + mem[4])
    print(description[4] + " : " + mem[5])
    print(description[5] + " : " + mem[6])

    print()

    return temperature
