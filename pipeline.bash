#!/bin/sh

# this bash script assumes that you have already:
# downloaded the folder of openpose json keypoints
# run pose_to_images to generate images
# and created targets.txt

# it assumes the following directory format
# {video_id}/                                       # from google colab
#            {video_id}.mp3                         # from google colab
#            {video_id}_targets.txt                 # manual
#            {video_id}_fps.txt                     # obtained by running pose_to_image.py
#            {video_id}_outputs/                    #  name is specified by user, but preferably follow {video_id} convention

# This script will:
# compile according to {video_id}_targets.txt
# preprocess
# make accompanying audio

home=$HOME
all_project_dir="$home/Desktop/final_project"
project_dir="$all_project_dir/w5BqAXwflW0"

json_input_dir="$project_dir/w5BqAXwflW0"
echo "$json_input_dir"

echo "Compiling according to $targets now..."
python3 compile_json_to_npy.py --input_dir $json_input_dir --output_dir $project_dir
echo "Compiling completed. Now preprocessing..."

python3 preprocess.py --input_dir $project_dir
echo "Preprocessing completed. Generating audio features..."

python3 audioToMFCC.py --input_dir $project_dir 



