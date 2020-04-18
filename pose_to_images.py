import os
import json
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import argparse

"""
draw_pose_figure adapted from:
https://github.com/xrenaa/Music-Dance-Video-Synthesis/blob/master/dataset/output_helper.py
"""

parser = argparse.ArgumentParser()

# 19 and 20 are chin and top of head
joint_to_limb_heatmap_relationship = [
    [1,8],   [1,2],   [1,5],   [2,3],   [3,4],
    [5,6],   [6,7],   [8,9],   [9,10],  [10,11],
    [8,12],  [12,13], [13,14], [1,0],   [0,15],
    [15,17], [0,16],  [16,18]]

colors = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 0], [255,0,0],
    [255,0,0], [255,0,0], [255,0,0], [255,0,0]]

def imshow(image):
    plt.imshow(image[:,:,[2,1,0]].astype(np.uint8))
    plt.show()

def label_canvas(frame_num, canvas):
    font = cv2.FONT_HERSHEY_SIMPLEX
    # put it top middle
    coord = (int(canvas.shape[0] / 2), 150)
    cv2.putText(canvas, str(frame_num),
                coord, font,
                3, (0, 0, 0), thickness=5)
    return canvas
    
def add_pose_to_canvas(person_id, coors, canvas, limb_thickness=4):
    limb_type = 0
    labeled = False
    for joint_relation in  joint_to_limb_heatmap_relationship: # [1, 8] for example
        joint_coords = coors[joint_relation] # use it as indices 
        if 0 in joint_coords: # if a keypoint has 0
            # ignore keypoints that are not predicted
            limb_type += 1
            continue
        for joint in joint_coords:  # Draw circles at every joint
            if not labeled:
                font = cv2.FONT_HERSHEY_SIMPLEX
                print(joint[0:2])
                cv2.putText(canvas, person_id,
                            tuple(joint[0:2].astype(int)), font,
                            2, (0,0,0), thickness=5)
                labeled = True
            cv2.circle(canvas, tuple(joint[0:2].astype(
                        int)), 4, (0,0,0), thickness=-1)
        coords_center = tuple(
                    np.round(np.mean(joint_coords, 0)).astype(int))
        limb_dir = joint_coords[0, :] - joint_coords[1, :]
        limb_length = np.linalg.norm(limb_dir)
        angle = math.degrees(math.atan2(limb_dir[1], limb_dir[0]))
        polygon = cv2.ellipse2Poly(
                    coords_center, (int(limb_length / 2), limb_thickness), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, colors[limb_type])
        limb_type += 1
    return canvas

    
def draw_pose_figure(person_id, coors, height=1500, width=1500, limb_thickness=4):
    canvas = np.ones([height,width,3])*255
    canvas = canvas.astype(np.uint8)
    limb_type = 0
    return add_pose_to_canvas(person_id, coors, canvas, limb_thickness)


parser.add_argument("--input_dir",
                    help="Path to directory containing json keypoints",
                    default='/Users/will.i.liam/Desktop/final_project/jardy/outputVEE5qqDPVGY/',
                    type=str)
parser.add_argument("--output_dir",
                    help="Path to output directory containing images",
                    default='/Users/will.i.liam/Desktop/final_project/jardy/images_normed/',
                    type=str)

args = parser.parse_args()

def check_dir(output_dir):
    # check if output_dir exists
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        # check if there are files in the directory already
        if os.listdir(output_dir):
            # delete files
            import shutil
            try:
                shutil.rmtree(output_dir)
            except OSError as e:
                print("Error: %s : %s" % (output_dir, e.strerror))
            os.mkdir(output_dir)
    else:
        os.mkdir(output_dir)

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

check_dir(args.output_dir)

json_files = [pos_json for pos_json in os.listdir(args.input_dir) if pos_json.endswith('.json')]
json_files = sorted(json_files)

shift = np.array([0, 0])

frame = 0
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
