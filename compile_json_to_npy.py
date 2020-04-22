import os
import json
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import argparse
import sys
from data_utils.check_dirs import check_input_dir, check_output_dir
from data_utils.read_desired_frames import read_desired_frames

def get_json_names(input_dir):
    """
    Go into input dir and obtain all json file names
    """
    # confirm input dir is valid and modify input dir if it doesn't end in /
    input_dir = check_input_dir(input_dir)
    json_file_names = [pos_json for pos_json in os.listdir(input_dir) if pos_json.endswith('.json')]
    json_file_names = sorted(json_file_names)
    return json_file_names

def compile_all_poses(input_dir, output_dir, desired_person_at_frame):
    """
    Combines all json files into one massive json according to person_per_frame

    NOTE: User is responsible for pre-screening data and selecting person across all specified frames
    Thus, expected that there should be no missing poses in the data

    desired_person_at_frame is an array of tuples
    where id is an int, 
    frame_begin is beginning of wanted frames inclusive,
    frame_end is end of wanted frames also inclusive
    """

    # create a dictionary where key = index of frame we're on, with two values person_id and 
    all_poses = []
    
    json_files = get_json_names(input_dir)

    # iterate through each pose, frame_begin, frame_end tuple
    count = 0
    for target_person_id, frame_begin, frame_end in desired_person_at_frame:
        # go through the targeted ones
        for json_file in json_files[frame_begin:frame_end + 1]:
            data = json.load(open(input_dir + json_file))
            # look for the correct person among all the people in data['people']
            for index, person in enumerate(data['people']):
                if person['person_id'][0] == target_person_id:
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
                    all_poses.append(np_keypoints)
        # save the file
        # eventually we're gonna need a biggg dataset
        np.save(output_dir + 'compiled_data_line_' + str(count) + '.npy', np.array(all_poses))
        # reset
        all_poses = []
        count += 1

def main(args):
    input_dir = check_input_dir(args.input_dir)
    output_dir = check_output_dir(args.output_dir)
    desired_person_at_frame = read_desired_frames(args.targets)
    compile_all_poses(input_dir, output_dir, desired_person_at_frame)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",
                    help="Path to directory containing json keypoints",
                    default='/Users/will.i.liam/Desktop/final_project/VEE5qqDPVGY/outputVEE5qqDPVGY/',
                    type=str)
    parser.add_argument("--output_dir",
                        help="Path to desired output directory to save data file to",
                        default='/Users/will.i.liam/Desktop/final_project/VEE5qqDPVGY/data/',
                        type=str)

    parser.add_argument('--targets',
                        help="Direct me to list of target per frame",
                        default="/Users/will.i.liam/Desktop/final_project/VEE5qqDPVGY/targets.txt",
                        type=str)
    args = parser.parse_args()
    main(args)
