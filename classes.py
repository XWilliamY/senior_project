import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class AudioToPosesDataset(Dataset):
    """ Aligns mfcc .npy file to poses .npy file """
    def __init__(self, mfcc_file=None, pose_file=None, seq_len=1):
        """
        If given just an mfcc_file, configure data file for testing
        Otherwise, creates audio_video pairs

        Args:
             mfcc_file (string): Path to .npy file w/ mfccs
             pose_file (optional, string): Path to .npy file w/ joint keypoints
        """
        self.mfcc_file = mfcc_file
        self.pose_file = pose_file 
        self.seq_len = seq_len
    
        self.mfccs = np.load(self.mfcc_file)
        # transform to total_frames by mfcc_features
        self.mfccs = self.mfccs.T.astype('float64')
            
        if self.pose_file:
            self.poses = np.load(self.pose_file).astype('float64')
        else: # create dummy
            self.poses = np.zeros([int(np.floor(self.mfccs.shape[0] / 3)),
                                   19,
                                   2])


    def hasPoses(self):
        if self.pose_file:
            return True
        return False
    
    def getDataDims(self):
        """
        MFCC dimensions
        Pose dimensions
        """
        return self.mfccs.shape, self.poses.shape

    def getSingleInputFeatureDims(self):
        return self.mfccs.shape[-1] * 3

    def getSingleOutputFeatureDims(self):
        return self.poses.shape[1] * self.poses.shape[2]
    
    def getDimsPerBatch(self):
        return self.getSingleInputFeatureDims(), self.getSingleOutputFeatureDims()
    
    def __len__(self):
        """
        Returns the number of video frames
        """
        return int( len(self.poses) / self.seq_len)

    def __getitem__(self, idx):
        """
        Args:
             idx (int): Assumes video frame

        Returns:
                X   (array): mfccs of shape [seq_len, mfcc_features]
                y   (array): keypoints of shape [joints, 2]
        """

        mfcc_idx = self.seq_len * 3 # 3 mfcc frames per timestep
        X = self.mfccs[idx * mfcc_idx : (idx + 1) * mfcc_idx, :]
        X = X.reshape([self.seq_len, -1])
        y = self.poses[idx * self.seq_len : (idx+1) * self.seq_len]
        y = y.reshape([self.seq_len, -1])
        return X, y

class PosesToPosesDataset(Dataset):
    """ Aligns mfcc .npy file to poses .npy file """
    def __init__(self, poses_file=None, seq_len=1):

        """
        If given just an mfcc_file, configure data file for testing
        Otherwise, creates audio_video pairs

        Args:
             input_poses (string): Path to input poses
             output_poses (string): Path to output poses
        """
        self.seq_len = seq_len

        # offset them by seq_len
        self.input_poses = np.load(poses_file)[:-1]
        self.output_poses = np.load(poses_file)[1:]
    
    def getDataDims(self):
        return self.input_poses.shape, self.output_poses.shape

    def getSingleInputFeatureDims(self):
        return self.input_poses.shape[1] * self.input_poses.shape[2]

    def getSingleOutputFeatureDims(self):
        return self.output_poses.shape[1] * self.output_poses.shape[2]
    
    def getDimsPerBatch(self):
        return self.getSingleInputFeatureDims(), self.getSingleOutputFeatureDims()
    
    def __len__(self):
        """
        Returns the number of video frames
        """
        return int( len(self.output_poses) / self.seq_len)

    def __getitem__(self, idx):
        """
        Args:
             idx (int): Assumes video frame

        Returns:
                X   (array): mfccs of shape [seq_len, mfcc_features]
                y   (array): keypoints of shape [joints, 2]
        """

        X = self.input_poses[idx * self.seq_len : (idx+1) * self.seq_len]
        X = X.reshape([self.seq_len, -1])
        y = self.output_poses[idx * self.seq_len : (idx+1) * self.seq_len]
        y = y.reshape([self.seq_len, -1])
        return X, y

