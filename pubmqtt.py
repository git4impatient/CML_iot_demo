#!/usr/bin/env python
"""a simple sensor data generator that sends to an MQTT broker via paho"""
import sys
import json
import time
import random

import paho.mqtt.client as mqtt
for line in sys.stdin:
        line = line.strip()
	mqttc = mqtt.Client()
	mqttc.connect('localhost', 1883)
	mqttc.publish('iot', line)

