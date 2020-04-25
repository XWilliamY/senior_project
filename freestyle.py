import torch
from torch import optim
from torch.autograd import Variable
from torch.utils import data
from classes import AudioToPosesDataset
import matplotlib.pyplot as plt
import os
import argparse
import logging
import numpy as np
import json

'''
This script takes an input of audio MFCC features and uses
an LSTM recurrent neural network to learn to predict
body joints coordinates
'''

logging.basicConfig()
log = logging.getLogger("mannerisms_rnn")
log.setLevel(logging.DEBUG)
torch.manual_seed(1234)
np.random.seed(1234)


class AudioToBodyDynamics(object):
    """
    Defines a wrapper class for training and evaluating a model. 
    Inputs:
           args    (argparse object):      model settings
           dataset (pytorch Dataloader):   DataLoader wrapper around Dataset
    """

    def __init__(self, args, generators, is_test=False, freestyle=False):
        # TODO
        super(AudioToBodyDynamics, self).__init__()
        self.device = args.device
        self.log_frequency = args.log_frequency

        self.is_test_mode = is_test
        self.is_freestyle_mode = freestyle
        self.train_generator = generators[0]
        self.val_generator = generators[1]
        self.model_name = args.model_name

        model_options = {
            'device': args.device,
            'dropout': args.dp,
            'batch_size': args.batch_size,
            'hidden_dim': args.hidden_size,
            'input_dim': args.input_dim,
            'output_dim': args.output_dim,
            'trainable_init': args.trainable_init
        }

        if args.model_name == "AudioToJointsThree":
            from model import AudioToJointsThree
            self.model = AudioToJointsThree(model_options).cuda(args.device)
        elif args.model_name == "AudioToJoints":
            from model import AudioToJoints
            self.model = AudioToJoints(model_options).cuda(args.device)
            

        # construct the model
        self.optim = optim.Adam(self.model.parameters(), lr=args.lr)

        # Load checkpoint model
        if self.is_freestyle_mode:
            self.loadModelCheckpoint(args.test_model)

    # loss function
    def buildLoss(self, predictions, targets):
        square_diff = (predictions - targets)**2
        out = torch.sum(square_diff, 1, keepdim=True)
        return torch.mean(out)

    def saveModel(self, state_info, path):
        torch.save(state_info, path)

    def loadModelCheckpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optim_state_dict'])

    def runNetwork(self, inputs, targets, validate=False):
        """
        Train on one given mfcc pose pair
        Args:
             inputs (array): [batch, seq_len, mfcc_features * 3]
             targets (array): [batch, seq_len, 19 * 2 poses]
        Returns:
             predictions, truth, loss
        """

        def to_numpy(x):
            # import from gpu device to cpu, convert to numpy
            return x.cpu().data.numpy()

        predictions = self.model.forward(inputs)

        # Get loss in MSE of pose coordinates
        loss = self.buildLoss(predictions, targets)

        # # Get loss in pixel space
        return (to_numpy(predictions), to_numpy(targets)), loss

    def runEpoch(self):
        # TODO
        train_losses = []
        val_losses = []
        predictions, targets = [], []

        if not self.is_freestyle_mode:
            for mfccs, poses in self.generator:
                self.model.train() # pass train flag to model
                # mfccs = mfccs.float()
                # poses = poses.float()

                vis_data, train_loss = self.runNetwork(mfccs, poses,
                                                validate=False)
                self.optim.zero_grad()
                train_loss.backward()
                self.optim.step()

                train_loss = train_loss.data.tolist()
                train_losses.append(train_loss)

                '''
                print('='*80)
                print("in runEpoch")
                print('^the predicted')
                print(vis_data[0][0])
                print(vis_data[0][0].dtype)
                print('^^the actual')
                print(vis_data[1][0])
                print(vis_data[1][0].dtype)
                '''

        if self.is_freestyle_mode:
            print('freestyle mode')
            for mfccs, poses in self.generator:
                self.model.eval()
                # mfccs = mfccs.float()
                vis_data, val_loss = self.runNetwork(mfccs, poses,
                                                     validate=True)
                val_loss = val_loss.data.tolist()
                val_losses.append(val_loss)
                pred = vis_data[0].reshape(int(vis_data[0].shape[0] *
                                               vis_data[0].shape[1]),
                                           19,
                                           2)
                predictions.append(pred)
                targets.append(vis_data[1])
            
        return train_losses, val_losses, predictions, targets

    def trainModel(self, max_epochs, logfldr, patience):
        # TODO
        log.debug("Training model")
        epoch_losses = []
        batch_losses = []
        val_losses = []
        i, best_loss, iters_without_improvement = 0, float('inf'), 0
        best_train_loss, best_val_loss = float('inf'), float('inf')

        for i in range(max_epochs):
            iter_train, iter_val, predictions, targets = self.runEpoch()
            iter_mean = np.mean(iter_train[0]), np.mean(iter_train[1])
            # iter_val_mean = np.mean(iter_val[0]), np.mean(iter_val[1])

            epoch_losses.append(iter_mean)
            batch_losses.extend(iter_train)
            # val_losses.append(iter_val_mean)

            log.info("Epoch {} / {}".format(i, max_epochs))
            log.info("Training Loss (1980 x 1080): {}".format(iter_mean))
            best_train_loss = iter_mean
            # log.info("Validation Loss (1980 x 1080): {}".format(iter_val_mean))
            '''
            improved = iter_val_mean[1] < best_loss
            if improved:
                best_loss = iter_val_mean[1]
                best_val_loss = iter_val_mean
                best_train_loss = iter_mean
                iters_without_improvement = 0
            else:
                iters_without_improvement += 1
                if iters_without_improvement >= patience:
                    log.info("Stopping Early because no improvment in {}".format(
                        iters_without_improvement))
                    break
            # if improved or (i % self.log_frequency) == 0:
            #     # Save the model information
            #     path = os.path.join(logfldr, "Epoch_{}".format(i))
            #     os.makedirs(path)
            #     path = os.path.join(path, "model_db.pth")
            #     state_info = {
            #         'epoch': i,
            #         'epoch_losses': epoch_losses,
            #         'batch_losses': batch_losses,
            #         'validation_losses': val_losses,
            #         'model_state_dict': self.model.state_dict(),
            #         'optim_state_dict': self.optim.state_dict(),
            #         'data_state_dict': self.data_iterator.stateDict()
            #     }
            #     self.saveModel(state_info, path)
            #     if improved:
            #         path = os.path.join(logfldr, "best_model_db.pth")
            #         self.saveModel(state_info, path)
            #
            #     # Visualize the PCA Coefficients
            #     num_vis = min(3, targets[0].shape[-1])
            #     for j in range(num_vis):
            #         save_path = os.path.join(
            #             logfldr, "Epoch_{}/pca_{}.png".format(i, j))
            #         self.visualizePCA(predictions[0], targets[0], j, save_path)
            '''

        # self.plotResults(logfldr, epoch_losses, batch_losses, val_losses)
        # return best_train_loss, best_val_loss
        state_info = {
            'epoch': i,
            'epoch_losses': epoch_losses,
            'batch_losses': batch_losses,
            'validation_losses': val_losses,
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
        }
        path = "saved_models" + "/" + self.model_name + ".pth"
        self.saveModel(state_info, path)
        return best_train_loss

    # def formatVizArrays(self, predictions, targets):
    #     final_pred, final_targ = [], []
    #     for ind, pred in enumerate(predictions):
    #         pred = self.data_iterator.toPixelSpace(pred)
    #         targ = self.data_iterator.toPixelSpace(targets[ind])
    #         pred = self.data_iterator.reconstructKeypsOrder(pred)
    #         targ = self.data_iterator.reconstructKeypsOrder(targ)
    #         final_pred.append(pred)
    #         final_targ.append(targ)
    #
    #     final_pred, final_targ = np.vstack(final_pred), np.vstack(final_targ)
    #     final_pred = final_pred[0::(2**self.upsample_times)]
    #     final_targ = final_targ[0::(2**self.upsample_times)]
    #
    #     return final_pred, final_targ

    # def visualizePCA(self, preds, targets, pca_dim, save_path):
    #     preds = self.data_iterator.getPCASeq(preds, pca_dim=pca_dim)
    #     targs = self.data_iterator.getPCASeq(targets, pca_dim=pca_dim)
    #     assert(len(preds) == len(targs))
    #     plt.plot(preds, color='red', label='Predictions')
    #     plt.plot(targs, color='green', label='Ground Truth')
    #     plt.legend()
    #     plt.savefig(save_path)
    #     plt.close()

    def plotResults(self, logfldr, epoch_losses, batch_losses, val_losses):
        losses = [epoch_losses, batch_losses, val_losses]
        names = [
            ["Epoch pixel losses", "Epoch coeff losses"],
            ["Batch pixel losses", "Batch coeff losses"],
            ["Val pixel losses", "Val coeff losses"]]
        _, ax = plt.subplots(nrows=len(losses), ncols=2)
        for index, pair in enumerate(zip(losses, names)):
            for i in range(2):
                data = [pair[0][j][i] for j in range(len(pair[0]))]
                ax[index][i].plot(data, label=pair[1][i])
                ax[index][i].legend()
        save_filename = os.path.join(logfldr, "results.png")
        plt.savefig(save_filename)
        plt.close()


def createOptions():
    #TODO
    # Default configuration for PianoNet
    parser = argparse.ArgumentParser(
        description="Pytorch: Audio To Body Dynamics Model"
    )
    parser.add_argument('--model_name', type=str, default="AudioToJoints", required=True)
    # parser.add_argument("--data", type=str, default="piano_data.json",
    #                     help="Path to data file")

    # dior_pop_smoke.mp3',
    parser.add_argument("--audio_file", type=str, default='/Users/will.i.liam/Desktop/final_project/VEE5qqDPVGY/audio/dior_pop_smoke.mp3',
                        help="Only in for Test. Location audio file for generating test video")
    parser.add_argument("--freestyle", type=bool, default=False,
                        help="Expects an audio file. Does not take in any pose files: model will generate according to given audio.")
    parser.add_argument("--logfldr", type=str, default=None,
                        help="Path to folder to save training information")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Training batch size. Set to 1 in test")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="The fraction of the training data to use as validation")
    parser.add_argument("--hidden_size", type=int, default=1024,
                        help="Dimension of the hidden representation")
    parser.add_argument("--test_model", type=str, default=None,
                        help="Location for saved model to load")
    parser.add_argument("--visualize", type=bool, default=False,
                        help="Visualize the output of the model. Use only in Test")
    parser.add_argument("--save_predictions", type=bool, default=True,
                        help="Whether or not to save predictions. Use only in Test")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to train on. Use 'cpu' if to train on cpu.")
    parser.add_argument("--max_epochs", type=int, default=10,
                        help="max number of epochs to run for")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning Rate for optimizer")
    parser.add_argument("--time_steps", type=int, default=60,
                        help="Prediction Timesteps")
    parser.add_argument("--patience", type=int, default=100,
                        help="Number of epochs with no validation improvement"
                        " before stopping training")
    parser.add_argument("--time_delay", type=int, default=6,
                        help="Time delay for RNN. Negative values mean no delay."
                        "Give in terms of frames. 30 frames = 1 second.")
    parser.add_argument("--dp", type=float, default=0.1,
                        help="Dropout Ratio For Training")
    parser.add_argument("--numpca", type=int, default=15,
                        help="number of pca dimensions. Use -1 if no pca - "
                             "Train on XY coordinates")
    parser.add_argument("--log_frequency", type=int, default=10,
                        help="The frequency with which to checkpoint the model")
    parser.add_argument("--trainable_init", action='store_false',
                        help="LSTM initial state should be trained. Default is True")
    parser.add_argument("--input_dim")
    parser.add_argument("--output_dim")

    args = parser.parse_args()
    return args


def main():
    args = createOptions()
    args.device = torch.device(args.device)
    # data_loc = args.data
    is_test_mode = args.test_model is not None

    root_dir = 'data/'
    mfcc_file = ""
    pose_file = None
    seq_len = 3

    # determine whether freestyle or not
    if args.freestyle: # needs audio file
        if args.audio_file:
            # convert audio to mfcc
            from audioToMFCC import convert
            output_file_path = convert(args.audio_file, "", None)
            print(output_file_path)
            # get audio
            mfcc_file = output_file_path
            pose_file = None
        else:
            print("Missing audio file")
    else:
        mfcc_file = root_dir + 'VEE5qqDPVGY_210_9810_mfccs.npy'
        pose_file = root_dir + 'processed_compiled_data_line_0.npy'


    print(mfcc_file)
    print(pose_file)
    dataset = AudioToPosesDataset(mfcc_file, pose_file, seq_len)
    val_size = int(args.val_split * len(dataset))
    train_size = len(dataset) - val_size

    train, validate = data.random_split(dataset, [train_size, val_size])
    
    params = {'batch_size':10,
              'shuffle':False,
              'num_workers': 1
              }
    train_generator = data.DataLoader(train, **params)
    val_generator = data.DataLoader(validate, **params)
    
    args.input_dim, args.output_dim = dataset.getDimsPerBatch()
    # Create model
    dynamics_learner = AudioToBodyDynamics(args, 
                                           (train, validate), 
                                           freestyle=args.freestyle)

    # logfldr = args.logfldr
    # if not os.path.isdir(logfldr):
    #     os.makedirs(logfldr)

    # Train model
    if not args.freestyle:
        print("training")
        min_train = dynamics_learner.trainModel(
            args.max_epochs, args.logfldr, args.patience)
        best_losses = [min_train]
    else:
        outputs = dynamics_learner.runEpoch() # train_losses, val_loss, preds, targets
        iter_train, iter_val, preds, target = outputs
        # min_train, min_val = np.mean(iter_train[0]), np.mean(iter_val[0])
        # save the predictions
        best_losses = [iter_train]
        np_preds = np.vstack(preds)
        np.save("dior.npy", np_preds)
        print(np_preds.shape)
        print(np_preds.dtype)

    log.info("The best validation is : {}".format(best_losses))


if __name__ == '__main__':
    main()
