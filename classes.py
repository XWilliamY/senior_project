import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class AudioToPosesDataset(Dataset):
    """ Aligns mfcc .npy file to poses .npy file """

    def __init__(self, mfcc_file, pose_file, seq_len):
        """
        Args:
             mfcc_file (string): Path to .npy file w/ mfccs
             pose_file (string): Path to .npy file w/ joint keypoints
        """
        self.mfcc_file = mfcc_file
        self.pose_file = pose_file
        self.seq_len = seq_len

        self.mfccs = np.load(self.mfcc_file)
        # need to transform mfccs since it's currently
        # mfcc_features by total_frames
        self.mfccs = self.mfccs.T.astype('float64')
        self.poses = np.load(self.pose_file).astype('float64')

    def getDims(self):
        """
        MFCC dimensions
        Pose dimensions
        """
        
    
    def __len__(self):
        """
        Returns the number of video frames
        """
        return len(self.poses)

    def __getitem__(self, idx):
        """
        Args:
             idx (int): Assumes video frame

        Returns:
                X   (array): mfccs of shape [seq_len, mfcc_features]
                y   (array): keypoints of shape [joints, 2]

        self.seq_len = 3
        0 * self.seq_len = 0
        (0 + 1) * 3  * self.seq_len = 9

        1 * 3 = 9
        (1 + 1) * 3  * self.seq_len = 18
        """

        mfcc_idx = self.seq_len * 3 # 3 mfcc frames per timestep
        X = self.mfccs[idx * mfcc_idx : (idx + 1) * mfcc_idx, :]
        X = X.reshape([self.seq_len, -1]) 
        y = self.poses[idx, idx+self.seq_len]
        y = y.reshape([self.seq_len, -1])
        return X, y
        