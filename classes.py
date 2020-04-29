import torch
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader

class AudioToPosesDirDataset(Dataset):
    """
    Given a directory, AudioToPosesDirDataset will align all matching names of mfcc files to pose files.
    If pose2pose is given, we ignore audio files and create a dataset for autoencoders.
    If nextpose is given, we instead create training and target data from pose files, offset by one for pose generation
    """
    def __init__(self, directory=None, seq_len=1, pose2pose=False, nextpose=False):

        # handle all pose files first:
        # get names of all processed pose files
        self.processed_poses_names = glob.glob(directory+'processed*.npy')

        # load poses in memory
        self.processed_poses = np.array([np.load(name, mmap_mode='r') for name in self.processed_poses_names])

        # get pose frame lengths
        self.poses_lengths = np.array([pose.shape[0] for pose in self.processed_poses])

        # get total combined dataset len
        self.seq_len = seq_len
        self.dataset_lens = (self.poses_lengths / self.seq_len).astype(int)

        # get max index per dataset
        self.file_idx_limits = np.cumsum(self.dataset_lens)

        # handle inputs
        # if mfcc:
        # get names of all associated mfcc files in that order
        # processed_{video_id}_line_{line}
        # mfcc_{video_id}_line_{line} correspondingly
        self.pose2pose = pose2pose
        self.nextpose = nextpose
        self.audio2pose = not self.pose2pose and not self.nextpose

        if self.audio2pose:
            self.mfcc_names = []
            for name in self.processed_poses_names:
                pose_name = name.split('_')
                mfcc_name = directory + f"mfcc_{pose_name[-3]}_line_{pose_name[-1]}"
                self.mfcc_names.append(mfcc_name)

            self.mfccs = np.array([np.load(name, mmap_mode='r').T for name in self.mfcc_names])
            self.mfcc_lengths = np.array([mfcc.shape[0] for mfcc in self.mfccs])

        # only need to modify data for nextpose=True or both pose2pose and nextpose=False


    def __len__(self):
        return int(np.sum(self.dataset_lens))

    def __getitem__(self, idx):

        # handle targets first
        file_idx = np.argmax(self.file_idx_limits > idx)
        target_idx = idx - (self.file_idx_limits[file_idx] - self.dataset_lens[file_idx])
        # use file_idx and target_idx for both mfccs and pose
        target_pose = self.processed_poses[file_idx][target_idx * self.seq_len : (target_idx + 1) * self.seq_len]
        y = target_pose.reshape([self.seq_len, -1])
        copied_y = np.array(y)
        y = torch.from_numpy(copied_y)

        # handle source
        if self.audio2pose:
            mfcc_idx = self.seq_len * 3
            X = self.mfccs[file_idx][target_idx * mfcc_idx : (target_idx + 1) * mfcc_idx]
            X = X.reshape([self.seq_len, -1])
            copied_X = np.array(X)
            X = torch.from_numpy(copied_X)

        elif self.pose2pose:
            return y, y

        return X, y

    def getSingleInputFeatureDims(self):
        if self.audio2pose:
            return self.mfccs[0][-1].shape[0] * 3
        if self.pose2pose or self.nextpose:
            return self.getSingleOutputFeatureDims()

    def getSingleOutputFeatureDims(self):
        return self.processed_poses[0][-1].shape[0] * 2

    def getDimsPerBatch(self):
        return self.getSingleInputFeatureDims(), self.getSingleOutputFeatureDims()


class AudioToPosesTwoDataset(Dataset):
    """ Aligns mfcc .npy file to poses .npy file """
    def __init__(self, mfcc_file=None, pose_file=None):
        """
        If given just an mfcc_file, configure data file for testing
        Otherwise, creates audio_video pairs

        Args:
             mfcc_file (string): Path to .npy file w/ mfccs
             pose_file (optional, string): Path to .npy file w/ joint keypoints
        """
        self.mfcc_file = mfcc_file
        self.pose_file = pose_file

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
        return len(self.poses)

    def __getitem__(self, idx):
        """
        This will batch according to len of dataset
        Thus each idx corresponds to one video frame

        Args:
             idx (int): Assumes video frame

        Returns:
                X   (array): mfccs of shape [seq_len, mfcc_features]
                y   (array): keypoints of shape [joints, 2]
        """
        # list of indices
        np_idx = np.arange(idx[0] * 3, (idx[-1]+1) * 3)

        X = self.mfccs[np_idx]
        X = X.reshape([len(idx), -1])
        y = self.poses[idx]
        y = y.reshape([len(idx), -1])
        return X, y

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
