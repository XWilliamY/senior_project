import cv2
import numpy as np
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir",
                    help="Path to images",
                    default="/Users/will.i.liam/Desktop/final_project/jardy/images_normed/",
                    type=str)

parser.add_argument("--output_dir",
                    help="output path",
                    default="",
                    type=str)

parser.add_argument("--output_filename",
                    help="output filename",
                    default="jardy_normalized_shift_by_orig_neck.mp4",
                    type=str)
args = parser.parse_args()

img_array = []
filenames = sorted(glob.glob(args.input_dir + '*.jpg'))

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
