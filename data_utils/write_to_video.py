import numpy as np
import cv2
import argparse
from joints import joint_to_limb_heatmap_relationship, colors, imshow, label_canvas, add_pose_to_canvas, draw_pose_figure
from check_dirs import check_input_dir, check_output_dir


def np_to_image(processed_poses, output_dir):
    count = 0
    shift_by = np.array([750, 800]) - processed_poses[0][8]
    processed_poses += shift_by
    for pose in processed_poses:
        person_id = str(0) + ", " + str([0])
        canvas = draw_pose_figure(person_id, pose)
        file_name = output_dir + f"{count:05}.jpg"
        cv2.imwrite(file_name, canvas)
        count += 1
    
def np_to_video(processed_poses, output_dir):
    print(output_filename)

    interpolated_canvases = []
    count = 0
    for pose in processed_poses:
        if count == 1800:
            break
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

def main(args):
    input_file = args.input_file

    processed_poses = np.load(input_file)
    output_dir = check_output_dir(args.output_dir)
    
    np_to_image(processed_poses, output_dir)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str,
                        help="path to npy")
    parser.add_argument('--output_dir', type=str,
                        help="path to images")
    args = parser.parse_args()
    main(args)
    
