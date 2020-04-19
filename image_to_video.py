import cv2
import numpy as np
import glob
import argparse
from data_utils.check_dirs import check_input_dir, check_output_dir, check_mp4


def main(args):
    img_array = []
    input_dir = check_input_dir(args.input_dir)
    output_dir = check_output_dir(args.output_dir)
    mp4name = check_mp4(args.output_filename)
    
    filenames = sorted(glob.glob(input_dir + '*.jpg'))

    count = 0
    for filename in filenames:
        if count == 1800:
            break
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
        count += 1

    fourcc_format = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_filename, fourcc_format, 30, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",
                        help="Path to images",
                        default="/Users/will.i.liam/Desktop/final_project/jardy/images/",
                        type=str)

    parser.add_argument("--output_dir",
                        help="output path",
                        default="/Users/will.i.liam/Desktop/final_project/jardy/videos/",
                        type=str)
    
    parser.add_argument("--output_filename",
                        help="output filename",
                        default="jardy_no_modifications.mp4",
                        type=str)
    args = parser.parse_args()
    main(args)
