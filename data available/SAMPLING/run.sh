#!/bin/bash

source ~/.bashrc

conda activate da2-grasp

python da2_grasp.py -te 0 -m "u2f" -ss "[0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]" -T 50 -Nd 8
python da2_grasp.py -te 350 -m "u2f_reverse" -ss "[0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]" -T 50 -Nd 8

