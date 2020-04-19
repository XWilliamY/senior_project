import os
import json
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import argparse
import sys
from data_utils.check_dirs import check_input_dir, check_output_dir

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
    User is responsible for pre-screening data and selecting the correct person across all specified frames
    Therefore there should be no missing poses in the data

    The resulting json will then be cut up into X second pieces for training
    The array that is passed into compile_all_poses should be provided to the audio equivalent too
    So that the divisions are consistent and audio and pose can be paired up easily

    desired_person_at_frame expects at least one beginning and one end
    every even is beginning, every odd is end
    [(id, frame_begin, frame_end), ...]
    """

    # create a dictionary where key = index of frame we're on, with two values person_id and 
    all_poses = []
    
    json_files = get_json_names(input_dir)
    frame = 0
    
    for json_file in json_files:
        data = json.load(open(input_dir + json_file))

        add_to_json = False
        target_person_id = 0
        for pose_id, frame_begin, frame_end in desired_person_at_frame:
            if frame >= frame_begin and frame < frame_end: # within bounds
                add_to_json = True
                target_person_id = pose_id
                break
            
        if add_to_json:
            # look for the correct person
            for index, person in enumerate(data['people']):
                if person == target_person_id:
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
                    
                    person_id = str(index) + ", " + str(target_person_id)

                    all_poses.append(np_keypoints)

        frame += 1
    return np.array(all_poses)

def main(args):
    input_dir = check_input_dir(args.input_dir)
    output_dir = check_output_dir(args.output_dir)
    desired_person_at_frame = [(0, 153, 7878)]
    compiled_poses = compile_all_poses(input_dir, output_dir, desired_person_at_frame)
    np.save(output_dir + 'data.npy', compiled_poses)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",
                    help="Path to directory containing json keypoints",
                    default='/Users/will.i.liam/Desktop/final_project/jardy/outputVEE5qqDPVGY/',
                    type=str)
    parser.add_argument("--output_dir",
                        help="Path to output directory containing images",
                        default='/Users/will.i.liam/Desktop/final_project/jardy/compiled_json/',
                        type=str)
    
    args = parser.parse_args()
    main(args)
