import os
import json
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import argparse
import sys
from data_utils.check_dirs import check_input_dir, check_output_dir
from data_utils.joints import joint_to_limb_heatmap_relationship, colors, imshow, label_canvas, add_pose_to_canvas, draw_pose_figure

def normalize_in_frame(frame):
    """
    given openpose outputs, normalize coordinates for each keypoint, scaling by:
    shifting coordinates by neck coordinate and dividing all coordinates by trunk length
    trunk length calculated by distance between neck and mid-hip

    an estimated attempt to make each person's pose estimations proportional across frames


    expects frame in the order of openpose outputs
    [21, 2] where 2 corresponds to x- y- coordinates
    
    reduces the pixel coordinate size of each coordinate
    """
    trunk_length = np.sqrt(np.sum( (frame[8] - frame[1]) ** 2))
    if trunk_length > 0:
        return frame / trunk_length
    else:
        return frame

def normalize_and_shift(frame):
    neck = frame[1]
    subtracted = frame - neck
    return normalize_in_frame(subtracted)

def adjust_to_canvas(poses):
    poses = poses * 125
    return poses, np.ones([np.int(poses[8][1] * 2),
                    np.int(poses[8][0] * 2), 3]) * 255


def get_json_names(input_dir):
    """
    Go into input dir and obtain all json file names
    """
    # confirm input dir is valid and modify input dir if it doesn't end in /
    input_dir = check_input_dir(input_dir)
    json_file_names = [pos_json for pos_json in os.listdir(input_dir) if pos_json.endswith('.json')]
    json_file_names = sorted(json_file_names)
    return json_file_names

def draw_all_poses(input_dir, output_dir):
    """
    Converts each json file into a corresponding image
    """
    json_files = get_json_names(input_dir)
    frame = 0
    for json_file in json_files:
        data = json.load(open(input_dir + json_file))

        if len(data['people']) < 1:
            keypoints = [0] * 42 # uncomment this to create an image regardless of whether there are poses
            # continue
            person_id = str(-1) + ", " + str([-1])
            canvas = draw_pose_figure(person_id,
                                      np.array(keypoints).reshape(-1, 2))
        else:
            # turn each person's keypoints into an numpy array
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

        file_name = output_dir + json_file[-20:-15] + ".jpg"
        print(file_name)

        canvas = label_canvas(frame, canvas)
        # imshow(canvas)
        cv2.imwrite(file_name, canvas)
        frame += 1

def main(args):
    input_dir = check_input_dir(args.input_dir)
    output_dir = check_output_dir(args.output_dir)

    draw_all_poses(input_dir, output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",
                    help="Path to directory containing json keypoints",
                    default='/Users/will.i.liam/Desktop/final_project/VEE5qqDPVGY/outputVEE5qqDPVGY/',
                    type=str)
    parser.add_argument("--output_dir",
                        help="Path to output directory containing images",
                        default='/Users/will.i.liam/Desktop/final_project/VEE5qqDPVGY/images/',
                        type=str)
    
    args = parser.parse_args()
    main(args)
