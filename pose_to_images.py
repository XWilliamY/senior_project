import os
import json
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from dataset.output_helper import save_batch_images

# l ear, l eye, nose, r eye, r ear
# l wrist, l elbow, l shoulder, middle, r shoulder, r elbow, r wrist
# l hip, mid hip, r hip
# l knee, r knee
# l ankle, r ankle

joint_to_limb_heatmap_relationship = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10],
    [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
    [2, 16], [5, 17]]

joint_to_limb_heatmap_relationship = [
    [1,8],   [1,2],   [1,5],   [2,3],   [3,4],
    [5,6],   [6,7],   [8,9],   [9,10],  [10,11],
    [8,12],  [12,13], [13,14], [1,0],   [0,15],
    [15,17], [0,16],  [16,18]]

# 19 and 20 are chin and top of head

#for plot usage
colors = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 0], [255,0,0],
    [255,0,0], [255,0,0], [255,0,0], [255,0,0]]


def draw_pose_figure(coors,height=1500,width=1500,limb_thickness=4):
    canvas = np.ones([height,width,3])*255
    canvas = canvas.astype(np.uint8)
    limb_type = 0
    for joint_relation in  joint_to_limb_heatmap_relationship:
        '''
        if(limb_type >= 20):
            break
        '''
        joint_coords = coors[joint_relation]
        for joint in joint_coords:  # Draw circles at every joint
            #print('joint',joint)
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


def imshow(image):
    plt.imshow(image[:,:,[2,1,0]].astype(np.uint8))
    plt.show()

# cd content/output
path_to_json = '/Users/will.i.liam/Desktop/final_project/openpose/output/'
image_path = '/Users/will.i.liam/Desktop/final_project/phoan/images/'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
json_files = sorted(json_files)

prev_keypoints = [0] * 50
first_file = True
shift = np.array([0, 0])

for json_file in json_files:
    data = json.load(open(path_to_json + json_file))

    # shift everything else
    keypoints = []
    count = 0
    index = 0
    if len(data['people']) < 1:
        continue
    for keypoint in data['people'][0]['pose_keypoints_2d']:
        # if count is 2, ignore (the confidence)
        if count != 2:
            keypoints.append(keypoint)
            '''
            if keypoint == 0:
                keypoints.append(prev_keypoints[index])
            else:
                keypoints.append(keypoint)
            '''
            count += 1
            index += 1
        else:
            count = 0


    np_keypoints = np.array(keypoints).reshape(-1, 2)
    if first_file: # create shift-by based on first frame
        shift = np.array([1000, 1250] - np_keypoints[8])

    shifted = np_keypoints + shift

    file_name = image_path + json_file[:-15] + ".jpg"
    prev_keypoints = keypoints
    print(file_name)
    imshow(draw_pose_figure(shifted))
    # in their code, real is stacked vertically on top of fake
    # cv2.imwrite(file_name, draw_pose_figure(shifted))

    # imshow(draw_pose_figure(np_keypoints))
    first_file = False
