import torch
from torch.utils import data
from classes import AudioToPosesDataset, PosesToPosesDataset
from model import AudioToJoints
import torch.nn as nn

root_dir = 'data/'
mfcc_file = root_dir + 'VEE5qqDPVGY_210_9810_mfccs.npy'
pose_file = root_dir + 'processed_compiled_data_line_0.npy'
pose_file = None

seq_len = 9
dataset = AudioToPosesDataset(mfcc_file, pose_file, seq_len)

'''
print(dataset.hasPoses())
print('loaded dataset')
print(dataset.getDataDims())
print(dataset.getDimsPerBatch())
'''

params = {'batch_size':128,
          'shuffle':False,
          'num_workers': 1
          }



generator = data.DataLoader(dataset, **params)
print(generator.dataset.getDimsPerBatch())
for epoch in range(1):
    count = 0
    
    for mfccs, poses in generator:
        print(mfccs.shape, poses.shape, count)
        count += 1

        # model computations

pose_file = root_dir + 'processed_compiled_data_line_0.npy'
seq_len = 1
dataset = PosesToPosesDataset(pose_file, seq_len)
params = {'batch_size':1,
          'shuffle':False,
          'num_workers': 1
          }
generator = data.DataLoader(dataset, **params)
print(generator.dataset.getDimsPerBatch())
for epoch in range(1):
    count = 0
    for inputs, outputs in generator:
        print('='*80)
        print(inputs[0, 0, :10])
        print(outputs[0, 0, :10])
        count += 1
