import os
import json
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir",
                    help="Path to directory containing json keypoints",
                    default='/Users/will.i.liam/Desktop/final_project/openpose/output/',
                    type=str)
parser.add_argument("--output_dir",
                    help="Path to output directory containing images",
                    default='/Users/will.i.liam/Desktop/final_project/phoan/images/',
                    type=str)

args = parser.parse_args()


json_files = [pos_json for pos_json in os.listdir(args.input_dir) if pos_json.endswith('.json')]


json_files = sorted(json_files)



'''
keypoints = []
confidences = []
count = 0
# data has 21 but we only need 18, so keep that in mind
for index, person in enumerate(data['people']):
    for keypoint in person['pose_keypoints_2d']:
        if count != 2:
            keypoints.append(keypoint)
            count += 1
        else: # this is a confidence
            confidences.append(keypoint)
            count = 0
    break

# the distance we're given also wants confidences so

# simple sum of every squared value
'''
frame = 3125
data = json.load(open(args.input_dir + json_files[frame]))
person_1 = 0

a_keys_and_confs = np.array(data['people'][person_1]['pose_keypoints_2d']).reshape(-1, 3)
'''
print(a_keys_and_confs[:, 0])
print(a_keys_and_confs[:, 1])
print(a_keys_and_confs[:, 2])
'''


print("Person 1 ID")
print(data['people'][person_1]['person_id'])
data = json.load(open(args.input_dir + json_files[frame+1]))
person_2 = 0
b_keys_and_confs = np.array(data['people'][person_2]['pose_keypoints_2d']).reshape(-1, 3)
print("Person 2 ID")
print(data['people'][person_2]['person_id'])

"""
distance metric adapted from
https://arxiv.org/pdf/1811.12607.pdf
"""
def distance(key_a, key_b, conf_a, conf_b):
    confs = conf_a * conf_b > 0
    confs = confs.reshape(-1, 1)
    
    print(np.sum(((key_a - key_b)**2)*(confs)) / np.sum(confs))

distance(a_keys_and_confs[:, :2],
         b_keys_and_confs[:, :2],
         a_keys_and_confs[:, 2],
         b_keys_and_confs[:, 2])

