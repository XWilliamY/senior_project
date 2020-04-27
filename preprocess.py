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

def scale_to_30fps(compiled_poses, metadata):
    """
    Only handle 24, 25, or 30 fps
    """
    inserted = 0
    if metadata == 25:
        multiple = 5
    elif metadata == 24:
        multiple = 4
    elif metadata == 30:
        return compiled_poses
    else:
        return None

    for frame_idx in range(int(compiled_poses.shape[0]/multiple) * multiple)[::multiple]:
        insert_value = frame_idx + multiple + inserted
        copy_index = insert_value - 1
        compiled_poses = np.insert(compiled_poses, insert_value,
                                   compiled_poses[copy_index], 0)
        inserted += 1
    return compiled_poses
    

def preprocess_poses(compiled_poses, fps):
    """
    Takes in a pre-loaded, compiled numpy array of desired poses

    Applies simple filtering, interpolation, and smoothing
    """
    compiled_poses = scale_to_30fps(compiled_poses, fps)
    
    
    # filter, interpolate, and smooth
    for i in range(19): # evaluate per body part
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
    for i in range(len(interpolated_canvases)):
        if count == 3000:
            break
        out.write(interpolated_canvases[i])
        count += 1
    out.release()

def main(args):
    # if none, then get current working dir for input_dir
    if args.input_dir:
        input_dir = check_input_dir(args.input_dir)
    else: # assuming cwd
        input_dir = check_output_dir(args.input_dir)
    # default wil be same as input_dir

    # get frame rate
    input_path = input_dir.split('/')
    fps_txt = '/'.join(input_path) + input_path[-2] + "_fps.txt"
    fps = 30
    with open(fps_txt, 'r') as f:
        line = f.readline()
        fps = int(float(line.split()[0]))

    
    input_dir = check_output_dir(input_dir + 'data/')
    
    if not args.output_dir:
        output_dir = input_dir
    else:
        output_dir = check_output_dir(args.output_dir)

    # if specific input file is given, process only that one
    if args.input_file:
        input_filename = args.input_file.split('_')
        # load using input_dir
        compiled_poses = np.load(input_dir + args.input_file)
        processed = preprocess_poses(compiled_poses, fps)
        # compiled_{id}_line_{line}.npy
        filename = f"processed_{'_'.join(input_filename[1:])}"
        # save using output_dir
        print(f"Saved to {filename}")
        np.save(output_dir + filename, processed)
        print(processed.shape)
    else:
        # fetch every input file from input dir
        import glob
        for input_filename in glob.glob(input_dir + "compiled*.npy"):
            compiled_poses = np.load(input_filename)
            processed = preprocess_poses(compiled_poses, fps)
            input_filename = input_filename.split('/')[-1].split('_')
            output_filename = output_dir + f"processed_{'_'.join(input_filename[1:])}"
            print(f"Saved to {output_filename}")
            np.save(output_filename, processed)
            print(processed.shape)

        
    # generate video
    if args.write_video:
        write_to_video(processed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",
                        help="Path to directory containing relevant input data",                        
                        default=None,
                        type=str)

    parser.add_argument("--input_file",
                        help="Name of specific file to preprocess",
                        default=None,
                        type=str)
    
    parser.add_argument("--output_dir",
                        help="Path to output directory, default will be same as input_dir",
                        default=None,
                        type=str)

    parser.add_argument('--write_video',
                        default=False,
                        type=bool)

    parser.add_argument('--video_id',
                        default=None)
    
    args = parser.parse_args()
    main(args)
