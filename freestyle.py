import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils import data
from classes import AudioToPosesDataset, PosesToPosesDataset, AudioToPosesDirDataset
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
           generator (tuple DataLoader):   a tuple of at least one DataLoader
    """

    def __init__(self, args, generator, freestyle=False):
        # TODO
        super(AudioToBodyDynamics, self).__init__()
        self.device = args.device
        self.log_frequency = args.log_frequency

        self.is_freestyle_mode = freestyle

        self.generator = generator
        self.model_name = args.model_name
        self.ident = args.ident
        self.model_name = args.model_name

        input_dim, output_dim = generator[0].dataset.getDimsPerBatch()

        model_options = {
            'seq_len': args.seq_len,
            'device': args.device,
            'dropout': args.dp,
            'batch_size': args.batch_size,
            'hidden_dim': args.hidden_size,
            'input_dim': input_dim,
            'output_dim': output_dim,
            'trainable_init': args.trainable_init
        }

        if args.model_name == "AudioToJointsThree":
            from model import AudioToJointsThree
            self.model = AudioToJointsThree(model_options).cuda(args.device)
        elif args.model_name == 'AudioToJointsNonlinear':
            from model import AudioToJointsNonlinear
            self.model = AudioToJointsNonlinear(model_options).cuda(args.device)
        elif args.model_name == "AudioToJoints":
            from model import AudioToJoints
            self.model = AudioToJoints(model_options).cuda(args.device)
        elif args.model_name == 'JointsToJoints':
            from model import JointsToJoints
            self.model = JointsToJoints(model_options).cuda(args.device).double()
        elif args.model_name == 'LSTMToDense':
            from model import LSTMToDense
            self.model = LSTMToDense(model_options).cuda(args.device).double()
        elif args.model_name == 'AudioToJointsSeq2Seq':
            from model import AudioToJointsSeq2Seq
            self.model = AudioToJointsSeq2Seq(model_options).cuda(args.device).double()
        elif args.model_name == 'MDNRNN':
            from model import MDNRNN
            self.model = MDNRNN(model_options).cuda(args.device).double()
        elif args.model_name == 'VAE':
            from model import VAE
            self.model = VAE(model_options).cuda(args.device).double()


        # construct the model
        self.optim = optim.Adam(self.model.parameters(), lr=args.lr)

        # Load checkpoint model
        if self.is_freestyle_mode:
            path = "saved_models/" + args.model_name + str(args.ident) + '.pth'
            print(path)
            self.loadModelCheckpoint(path)

    # loss function
    def buildLoss(self, predictions, targets):
        print(predictions.shape)
        print(targets.shape)
        square_diff = (predictions - targets)**2
        out = torch.sum(square_diff, -1, keepdim=True)
        print(torch.mean(out))
        return torch.mean(out)

    def mdn_loss(self, y, pi, mu, sigma):
        m = torch.distributions.Normal(loc=mu, scale=sigma)
        loss = torch.exp(m.log_prob(y))
        loss = torch.sum(loss * pi, dim=2)
        loss = -torch.log(loss)
        return torch.mean(loss)

    # https://github.com/pytorch/examples/blob/master/vae/main.py, Appendix B of https://github.com/pytorch/examples/blob/master/vae/main.py
    def vae_loss(self, targets, recon_targets, mu, logvar):
        BCE = nn.functional.binary_cross_entropy(recon_targets, targets, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE+KLD

    def saveModel(self, state_info, path):
        torch.save(state_info, path)

    def loadModelCheckpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optim_state_dict'])

    def runNetwork(self, inputs, targets):
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


        inputs = Variable(torch.DoubleTensor(inputs.double()).to(self.device))

        # reshape targets into (batch * seq_len, input features)
        targets = Variable(torch.DoubleTensor(targets).to(self.device))

        if self.model_name == 'AudioToJointsSeq2Seq':
            predictions = self.model.forward(inputs, targets)
        elif self.model_name == 'VAE':
            predictions, mu, logvar = self.model.forward(inputs)
        else:
            predictions = self.model.forward(inputs)

        # Get loss in MSE of pose coordinates
        # loss = self.buildLoss(predictions, targets)

        criterion = nn.L1Loss()
        if self.model_name == 'AudioToJointsSeq2Seq':
            loss = criterion(predictions.to(self.device), targets.to(self.device).float())
        elif self.model_name == 'MDNRNN':
            # predictions = (pi, mu, sigma), (h, c)
            loss = self.mdn_loss(targets, predictions[0][0], predictions[0][1], predictions[0][2])
        elif self.model_name == 'VAE':
            loss = self.vae_loss(targets, predictions, mu, logvar)
        else:
            loss = criterion(predictions, targets)
        # # Get loss in pixel space
        return (to_numpy(predictions), to_numpy(targets)), loss

    def runEpoch(self):
        # given one epoch
        train_losses = [] #coeff_losses
        val_losses = []
        predictions, targets = [], []

        if not self.is_freestyle_mode: # train
            # for each data point
            count = 0
            for mfccs, poses in self.generator[0]:
                print(f"training {count}")
                self.model.train() # pass train flag to model

                pred_targs, train_loss = self.runNetwork(mfccs, poses)
                self.optim.zero_grad()
                train_loss.backward()
                self.optim.step()
                train_loss = train_loss.data.tolist()
                train_losses.append(train_loss)
                count += 1

            # validation loss
            count = 0
            for mfccs, poses in self.generator[1]:
                self.model.eval()
                print(f"eval {count}")
                pred_targs, val_loss = self.runNetwork(mfccs, poses)

                val_loss = val_loss.data.tolist()
                val_losses.append(val_loss)
                pred = pred_targs[0].reshape(int(pred_targs[0].shape[0] *
                                                 pred_targs[0].shape[1]),
                                             19,
                                             2)
                predictions.append(pred)
                targets.append(pred_targs[1])
                count += 1

        # test or predict / play w/ model
        if self.is_freestyle_mode:
            for mfccs, poses in self.generator[0]:
                self.model.eval()
                # mfccs = mfccs.float()
                pred_targs, val_loss = self.runNetwork(mfccs, poses)
                val_loss = val_loss.data.tolist()
                val_losses.append(val_loss)
                pred = pred_targs[0].reshape(int(pred_targs[0].shape[0] *
                                                 pred_targs[0].shape[1]),
                                             19,
                                             2)
                predictions.append(pred)
                targets.append(pred_targs[1])

        return train_losses, val_losses, predictions, targets

    def trainModel(self, max_epochs, logfldr, patience):
        # TODO
        log.debug("Training model")
        epoch_losses = []
        batch_losses = []
        val_losses = []
        i, best_loss, iters_without_improvement = 0, float('inf'), 0
        best_train_loss, best_val_loss = float('inf'), float('inf')

        filename = 'logs/epoch_of_model_' + str(self.ident) +'.txt'
        state_info = {
            'epoch': i,
            'epoch_losses': epoch_losses,
            'batch_losses': batch_losses,
            'validation_losses': val_losses,
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
        }

        for i in range(max_epochs):
            if int(i/10) == 0:
                if i == 0:
                    with open(filename, 'w') as f:
                        f.write(f"Epoch: {i} started\n")
                else:
                    with open(filename, 'a+') as f:
                        f.write(f"Epoch: {i} started\n")
                # save the model
                path = "saved_models/" + self.model_name + str(self.ident) + ".pth"
                self.saveModel(state_info, path)

            # train_info, val_info, predictions, targets
            iter_train, iter_val, predictions, targets = self.runEpoch()

            iter_mean = np.mean(iter_train)
            iter_val_mean = np.mean(iter_val)
            # iter_val_mean = np.mean(iter_val[0]), np.mean(iter_val[1])

            epoch_losses.append(iter_mean)
            batch_losses.extend(iter_train)
            val_losses.append(iter_val_mean)

            log.info("Epoch {} / {}".format(i, max_epochs))
            log.info(f"Training Loss : {iter_mean}")
            log.info(f"Validation Loss : {iter_val_mean}")
            best_train_loss = iter_mean if iter_mean < best_train_loss else best_train_loss

        # Visualize VAE latent space
        if self.model_name == 'VAE':
            x, y = [], []
            for input, output in self.generator:
                mu, logvar = self.model.encode(input)
                print("mu ", mu)
                print("logvar ", logvar)
                x.append(mu)
                y.append(logvar)
            plt.plot(x,y)
            plt.show()

        self.plotResults(logfldr, epoch_losses, batch_losses, val_losses)
        path = "saved_models/" + self.model_name + str(self.ident) + ".pth"
        self.saveModel(state_info, path)
        return best_train_loss

    def plotResults(self, logfldr, epoch_losses, batch_losses, val_losses):
        losses = [epoch_losses, batch_losses, val_losses]
        names = [
            ["Epoch loss"],
            ["Batch loss"],
            ["Val loss"]]
        _, ax = plt.subplots(nrows=len(losses), ncols=1)
        for index, pair in enumerate(zip(losses, names)):
            data = [pair[0][j] for j in range(len(pair[0]))]
                ax[index][i].plot(data, label=pair[1][i])
                ax[index][i].legend()
        save_filename = os.path.join(logfldr, "results.png")
        plt.savefig(save_filename)
        plt.close()

def createOptions():
    #TODO
    parser = argparse.ArgumentParser(
        description="Pytorch: Audio To Body Dynamics Model"
    )
    parser.add_argument('--p2p', type=bool, default=False)
    parser.add_argument('--model_name', type=str, default="AudioToJoints")
    parser.add_argument('--autoencode', type=bool, default=False)
    parser.add_argument("--audio_file", type=str, default=None,
                        help="Only in for Test. Location audio file for generating test video")
    parser.add_argument("--freestyle", type=bool, default=False,
                        help="Expects an audio file. Does not take in any pose files: model will generate according to given audio.")
    parser.add_argument("--logfldr", type=str, default=None,
                        help="Path to folder to save training information")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Training batch size. Set to 1 in test")
    parser.add_argument("--seq_len", type=int, default=3)
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="The fraction of the training data to use as validation")
    parser.add_argument("--hidden_size", type=int, default=1024,
                        help="Dimension of the hidden representation")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Location for saved model to load")
    # parser.add_argument("--visualize", type=bool, default=False,
    #                     help="Visualize the output of the model. Use only in Test")
    parser.add_argument("--device", type=str, default="cuda",
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
    parser.add_argument("--ident", type=int, help="Model identification")
    args = parser.parse_args()
    return args


def main():
    args = createOptions()
    args.device = torch.device(args.device)

    root_dir = 'data/'
    mfcc_file = ""
    pose_file = None
    seq_len = args.seq_len
    params = {'batch_size':args.batch_size,
              'shuffle':False,
              'num_workers': 1
              }

    if args.p2p:
        pose_file = root_dir + 'processed_compiled_data_line_0.npy'
        dataset = PosesToPosesDataset(pose_file, args.seq_len)
    else:
        # determine whether freestyle or not
        if args.freestyle: # needs audio file
            if args.audio_file:
                # convert audio to mfcc
                from audioToMFCC import convert
                output_file_path = convert(args.audio_file, "", None, 0)
                print(output_file_path)
                # get audio
                mfcc_file = output_file_path
                pose_file = None

                # use pose_file=None feature of dataset
                dataset = AudioToPosesDataset(mfcc_file, pose_file, seq_len)
                generator = (data.DataLoader(dataset, **params))
            else:
                print("Missing audio file")
        else:
            # is training

            # check val_split first
            if args.val_split >= 1:
                print("Val split cannot be whole or greater than dataset.")
                exit(1)
            train_split = 1 - args.val_split
            # dataset will take (1 - val_split) percent of each individual .npy
            train_dataset = AudioToPosesDirDataset(root_dir, seq_len, pose2pose=args.autoencode, end=train_split)
            val_dataset = AudioToPosesDirDataset(root_dir, seq_len, pose2pose=args.autoencode, start=train_split)
            generator = (data.DataLoader(train_dataset), data.DataLoader(val_dataset))

    # Create model
    dynamics_learner = AudioToBodyDynamics(args,
                                           generator,
                                           freestyle=args.freestyle)

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
        np.save("pose_gang.npy", np_preds)

    log.info("The best validation is : {}".format(best_losses))


if __name__ == '__main__':
    main()
