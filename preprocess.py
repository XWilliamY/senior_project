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
from scipy.signal import medfilt#2d

def preprocess_poses(compiled_poses):
    """
    Takes in a pre-loaded, compiled numpy array of desired poses

    Applies simple filtering, interpolation, and smoothing
    """

    inserted = 0
    multiple = 4

    for frame_idx in range(int(compiled_poses.shape[0]/multiple) * multiple)[::multiple]:
        print('='*80)
        print(frame_idx)
        insert_value = frame_idx + multiple + inserted
        print(insert_value)
        copy_index = insert_value - 1
        print(copy_index)
        compiled_poses = np.insert(compiled_poses, insert_value,
                                   compiled_poses[copy_index], 0)
        inserted += 1
    print(compiled_poses.shape)

    # filter, interpolate, and smooth
    for i in range(21): # evaluate per body part
        # x or y coordinate
        for j in range(2):
            a_slice = compiled_poses[:, i, j]
            orig_shape = a_slice.shape
            a_slice = a_slice.reshape(-1)
        
            frames = np.where(a_slice != 0)[0]
            interpolated = np.interp(np.arange(orig_shape[0]), frames, a_slice[frames])
            med_filtered_slice = medfilt(interpolated)
            smoothed_slice = cv2.GaussianBlur(med_filtered_slice, (7, 7), 100)
            compiled_poses[:, i, j] = smoothed_slice.reshape(orig_shape)

    print(compiled_poses.shape)
    # if not 30 fps, need to duplicate some frames
    # assuming 24 fps, then every four frames, duplicate
    return compiled_poses

def write_to_video(processed_poses):
    interpolated_canvases = []
    count = 0
    for pose in processed_poses:
        if count == 3000:
            break
        person_id = str(0) + ", " + str([0])
        interpolated_canvases.append(draw_pose_figure(person_id,
                                                      pose))
        count += 1
    
    output_filename = 'upsampled_then_processed.mp4'
    height, width, layers = interpolated_canvases[0].shape
    size = (width,height)
    
    fourcc_format = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc_format, 30, size)

    count = 0
    print("Creating video")
    for i in range(len(interpolated_canvases)):
        if count == 3000:
            break
        out.write(interpolated_canvases[i])
        count += 1
    out.release()

def main(args):
    input_dir = check_input_dir(args.input_dir)
    output_dir = check_output_dir(args.output_dir)
    compiled_poses = np.load(input_dir + args.input_file)

    # copy the original to a new npy
    processed = np.copy(compiled_poses)
    # process it
    processed = preprocess_poses(processed)
    # save file
    np.save(output_dir + "upsampled_24_to_30_then_processed" + args.input_file, processed)
    # generate video
    write_to_video(processed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",
                        help="Path to directory containing relevant npy data",                        
                        default='/Users/will.i.liam/Desktop/final_project/jardy/compiled_npy/',
                        type=str)

    parser.add_argument("--input_file",
                        help="Name of specific file to preprocess",
                        default='data.npy',
                        type=str)
    
    parser.add_argument("--output_dir",
                        help="Path to output directory, default will be same as input_dir",
                        default='/Users/will.i.liam/Desktop/final_project/jardy/compiled_npy/',
                        type=str)
    
    args = parser.parse_args()
    main(args)
