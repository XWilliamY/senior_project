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

def compile_all_poses(input_dir, output_dir, desired_person_at_frame, video_id):
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
    line = 0
    for target_person_id, frame_begin, frame_end in desired_person_at_frame:
        # go through the targeted ones
        for json_file in json_files[frame_begin:frame_end + 1]:
            data = json.load(open(input_dir + json_file))
            # look for the correct person among all the people in data['people']
            for index, person in enumerate(data['people']):
                if person['person_id'][0] == target_person_id:
                    keypoints = []
                    count = 0
                    for keypoint in person['pose_keypoints_2d'][:57]:
                        if count != 2:
                            keypoints.append(keypoint)
                            count += 1
                        else:
                            count = 0
                    np_keypoints = np.array(keypoints).reshape(-1, 2)
                    all_poses.append(np_keypoints)
        # save the file
        filename = f"{output_dir}compiled_{video_id}_line_{line}.npy"
        npy_save_file = filename
        np.save(npy_save_file, np.array(all_poses))
        
        # reset
        all_poses = []
        line += 1

def main(args):
    input_dir = check_input_dir(args.input_dir)
    output_dir = check_output_dir(args.output_dir)

    output_path = output_dir.split('/')

    if args.video_id:
        video_id = args.video_id
    else:
        video_id = output_path[-2]
    frame_rate = 30
    fps_txt = '/'.join(output_path) + output_path[-2] + "_fps.txt"
    with open(fps_txt, 'r') as f:
        line = f.readline()
        fps = int(float(line.split()[0]))
    target_frame_rate = 30
    targets = '/'.join(output_path) + output_path[-2] + "_targets.txt"
    desired_person_at_frame = read_desired_frames(targets, frame_rate, target_frame_rate)

    # append '/data/' to output dir
    data_output_dir = check_output_dir(output_dir + 'data/')
    compile_all_poses(input_dir, data_output_dir, desired_person_at_frame, video_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",
                        help="Path to directory containing json keypoints",
                        default='/Users/will.i.liam/Desktop/final_project/w5BqAXwflW0/keypoints',
                        type=str)
    parser.add_argument("--output_dir",
                        help="Path to desired output directory to save data relevant files to. A data directory will be created in the provided location",
                        type=str)
    parser.add_argument("--video_id",
                        default=None)

    args = parser.parse_args()
    main(args)
