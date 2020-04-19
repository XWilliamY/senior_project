import os
import json
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import argparse
import pickle

"""
Converts a directory of json files into one large numpy array
"""
parser.add_argument("--input_dir",
                    help="Path to directory containing json keypoints",
                    default='/Users/will.i.liam/Desktop/final_project/jardy/outputVEE5qqDPVGY/',
                    type=str)
parser.add_argument("--output_path",
                    help="Path to output directory containing images",
                    default='/Users/will.i.liam/Desktop/final_project/jardy/images_normed/',
                    type=str)

parser.add_argument("--output_name",
                    default='compiled_json.pkl',
                    type=str)


args = parser.parse_args()

def check_dir(output_dir):
    # check if output_dir exists
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        return True
    else:
        return False

json_files = [pos_json for pos_json in os.listdir(args.input_dir) if pos_json.endswith('.json')]
json_files = sorted(json_files)

# (frame, num_ppl, body_parts, coordinates and confidences)
# person['person_id'] is more important since we can just look for that instead of the way we indexed

for json_file in json_files:
    data = json.load(open(args.input_dir + json_file))
    # delete this when no longer needed
    if frame == 1800:
        break
    # shift everything else
    '''
    keypoints = []
    count = 0
    '''
    if len(data['people']) < 1:
        # keypoints = [0] * 42
        continue
    else:
        for index, person in enumerate(data['people']):
            keypoints = []
            count = 0
            for keypoint in person['pose_keypoints_2d']:
                if count != 2:
                    keypoints.append(keypoint)
                    count += 1
                else:
                    count = 0
            # pass person['person_id'] and keypoints to canvas
            np_keypoints = np.array(keypoints).reshape(-1, 2)


            """
            normalized = normalize_in_frame(np_keypoints)
            neck = np_keypoints[0]
            np_keypoints = normalized * 125
            new_neck = np_keypoints[0]
            diff = neck - new_neck
            np_keypoints = np_keypoints + diff
            """

            # normalized = normalize_and_shift(np_keypoints)
            # np_keypoints = normalized * 125 + 500
            
            person_id = str(index) + ", " + str(person['person_id'])
            if index == 0:
                # create the canvas
                canvas = draw_pose_figure(person_id,
                                 np_keypoints)
                index += 1
            else:
                # add to the canvas
                add_pose_to_canvas(person_id,
                                   np_keypoints,
                                   canvas)
    '''
    # this was saving only one pose per frame
    else:
        for keypoint in data['people'][0]['pose_keypoints_2d']:
            # if count is 2, ignore (the confidence)
            if count != 2:
                keypoints.append(keypoint)
                count += 1
            else:
                count = 0


    np_keypoints = np.array(keypoints).reshape(-1, 2)
    if frame == 0: # create shift-by based on first frame
        shift = np.array([1000, 1250] - np_keypoints[8])

    shifted = np_keypoints + shift
    '''
    file_name = args.output_dir + json_file[-20:-15] + ".jpg"
    print(file_name)

    # add frame number to canvas
    canvas = label_canvas(frame, canvas)
    # imshow(canvas)
    cv2.imwrite(file_name, canvas)
    # imshow(draw_pose_figure(1, np_keypoints))
    # cv2.imwrite(file_name, draw_pose_figure(shifted))

    # imshow(draw_pose_figure(np_keypoints))
    frame += 1
