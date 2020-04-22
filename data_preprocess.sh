#!/bin/bash

# You should set up something like:
# umbrella_directory/{YOUTUBE_ID}/
# under {YOUTUBE_ID} there should be metadata.txt and outputs_{YOUTUBE_ID}

# pose to images
# manually create targets.txt in
# umbrella_directory/{YOUTUBE_ID}/

# compile json to npy
# by default saves to umbrella/{YOUTUBE_ID}/data/ as .npy

# preprocess
# uses umbrella/{YOUTUBE_ID}/metadata.txt really just needs fps
# creates processed data file and saves to umbrella/{YOUTUBE_ID}/data/ ...

# combine
# uses metadata and targets

# to index into audio, need sampling rate * frames / fps
# need the original, so current frame out of desired frames / fps
# numerator is from targets.txt, denominator is from metadata

