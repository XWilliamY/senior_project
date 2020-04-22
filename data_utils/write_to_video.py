import numpy as np
import cv2
from joints import joint_to_limb_heatmap_relationship, colors, imshow, label_canvas, add_pose_to_canvas, draw_pose_figure

def np_to_video(processed_poses, filename):
    chunked = filename.split('/')
    output_filename = '/'.join(chunked[:-2]) + '/videos/' + chunked[-1][:-4] + '_last.mp4'
    print(output_filename)

    interpolated_canvases = []
    count = 0
    for pose in processed_poses:
        if count < 7200:
            count += 1
            continue
        #break
        person_id = str(0) + ", " + str([0])
        interpolated_canvases.append(draw_pose_figure(person_id,
                                                      pose))
        count += 1

    # assumes .npy and proper location!
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

def file_to_video(filename):
    # expects .npy file
    poses = np.load(filename)
    np_to_video(poses, filename)

file_to_video('/Users/will.i.liam/Desktop/final_project/VEE5qqDPVGY/data/processed_compiled_data_line_0.npy')
