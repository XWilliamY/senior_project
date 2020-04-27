import torch
from torch.utils import data
from classes import AudioToPosesDataset, PosesToPosesDataset, AudioToPosesTwoDataset, AudioToPosesDirDataset
from model import AudioToJoints
import torch.nn as nn
import numpy as np

root_dir = 'data/'
seq_len = 30
dataset = AudioToPosesDirDataset(root_dir, seq_len)

params = {'batch_size':16,
          'shuffle':False,
          'num_workers': 1
          }


big = []
generator = data.DataLoader(dataset, **params)
for epoch in range(1):
    count = 0
    for mfcc, pose in generator:
        print(mfcc.shape)
        print(pose.shape)
        count += 1
print(f"{count} samples in total")
'''
small = []
mfcc_file = root_dir + 'mfcc_w5BqAXwflW0_line_0.npy'
pose_file = root_dir + 'processed_w5BqAXwflW0_line_0.npy'
dataset = AudioToPosesDataset(mfcc_file, pose_file, seq_len)
generator = data.DataLoader(dataset, **params)
for epoch in range(1):
    count = 0
    for mfcc, pose in generator:
        small.append(pose)
        count += 1


print('='*80)
print("first comparison")
print(f"{count} samples in small")
for pose in zip(small, big):
    print(np.array_equal(pose[0], pose[1]))


small = []
mfcc_file = root_dir + 'mfcc_w5BqAXwflW0_line_1.npy'
pose_file = root_dir + 'processed_w5BqAXwflW0_line_1.npy'
dataset = AudioToPosesDataset(mfcc_file, pose_file, seq_len)
generator = data.DataLoader(dataset, **params)
for epoch in range(1):
    count = 0
    for mfcc, pose in generator:
        small.append(pose)
        count += 1

print('='*80)
print("second comparison")
print(f"{count} samples in small")
for pose in zip(small, big[870:]):
    print(np.array_equal(pose[0], pose[1]))
'''
