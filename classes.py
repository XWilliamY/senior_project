import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class AudioToPosesDataset(Dataset):
    """ Aligns mfcc .npy file to poses .npy file """

    def __init__(self, mfcc_file, pose_file):
        """
        Args:
             mfcc_file (string): Path to .npy file w/ mfccs
             pose_file (string): Path to .npy file w/ joint keypoints
        """
        self.mfcc_file = mfcc_file
        self.pose_file = pose_file

        self.mfccs = np.load(self.mfcc_file)
        # need to transform mfccs since it's currently
        # mfcc_features by total_frames
        self.mfccs = self.mfccs.T.astype('float64')
        self.poses = np.load(self.pose_file).astype('float64')

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
                X   (array): mfccs of shape [3, mfcc_frames]
                y   (array): keypoints of shape [joints, 2]
        """

        X = self.mfccs[idx : idx + 3, :]
        y = self.poses[idx]
        return X, y
        
