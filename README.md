# Joint Senior Project of Ellie Shang and William Yang

## pose_to_images.py

Right now, the folder of pose estimations is hard-coded in the file.

Given pose estimation outputs from openpose, convert the 25 joint matrices into a skeleton model in a 2000 x 2000 grid.

Currently, pose estimation data is not scaled - ideally for training (and other tasks), such pose estimation data can be normalized to account for differences in dancers' body shapes and sizes, etc.

## image_to_video.py

Stitches a series of images into a video.avi - next steps involve incorporating audio, although this can also be done w/ ffmpeg.

