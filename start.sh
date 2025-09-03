#!/bin/bash
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless
pip install opencv-python-headless==4.8.0.74
python esp32cam.py
