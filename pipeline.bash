#!/bin/sh

# assuming you are starting from created targets.txt

# download folders
# run pose_to_images

# create targets.txt

# assuming {video_id} folder
# contains {images}, targets.txt, {json_folder}

# compile
# preprocess
# make accompanying audio

# for compile:
# need the inputs_dir of json
# also need a dir to save the data to

home=$HOME
# echo "$home"
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



