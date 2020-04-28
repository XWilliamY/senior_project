# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np


class Residual(nn.Module):
    def __init__(self, linear_hidden = 1024, time = 1024):
        super(Residual, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(linear_hidden, linear_hidden),
            nn.BatchNorm1d(time),
            nn.ReLU(),
            nn.Linear(linear_hidden, linear_hidden),
            nn.BatchNorm1d(time),
            nn.ReLU()
            )
    def forward(self, inputs):
        output = self.layer(inputs)
        return output
        
class JointsToJoints(nn.Module):
    def __init__(self, options):
        super(JointsToJoints, self).__init__()

        # linear 1
        self.options = options
        self.linear_encode = nn.Linear(self.options['input_dim'],
                                       self.options['hidden_dim'])
        # temporal batch normalization
        self.linear_encode_bn = nn.BatchNorm1d(self.options['hidden_dim'])
        self.linear_encode_tanh = nn.Tanh()
        self.dropout_encode = nn.Dropout(self.options['dropout'])

        # residuals
        self.res1 = Residual(linear_hidden = 1024)
        self.res2 = Residual(linear_hidden = 1024)
        self.dropout = nn.Dropout(p=0.5)
        self.linear_decode = nn.Linear(self.options['hidden_dim'],
                                       self.options['output_dim'])

    def forward(self, inputs):
        # flatten to (batch * seq_len, input dimensions)
        inputs = inputs.reshape(-1, self.options['input_dim'])

        # linear 1
        encoded = self.linear_encode(inputs)
        bn = self.linear_encode_bn(encoded)
        tanhed = self.linear_encode_tanh(bn)
        dropped = self.dropout_encode(tanhed)

        # residuals
        output_res1 = self.res1(dropped) + dropped
        output_res2 = self.res2(output_res1) + output_res1
        dropped = self.dropout(output_res2)
        decoded = self.linear_decode(dropped)

        predictions = decoded.reshape(-1,
                                      self.options['seq_len'],
                                      self.options['output_dim'])
        return predictions


        
class AudioToJointsThree(nn.Module):
    def __init__(self, options):
        super(AudioToJointsThree, self).__init__()
        self.init = None
        self.options = options
        self.lstm = nn.LSTM(self.options['input_dim'],
                            self.options['hidden_dim'],
                            batch_first=True,
                            num_layers=3).double()
        self.fc = nn.Linear(self.options['hidden_dim'],
                            self.options['output_dim']).double()
        self.initialize()

    def initialize(self):
        # Initialize LSTM Weights and Biases
        for layer in self.lstm._all_weights:
            for param_name in layer:
                if 'weight' in param_name:
                    weight = getattr(self.lstm, param_name)
                    init.xavier_normal_(weight.data)
                else:
                    bias = getattr(self.lstm, param_name)
                    init.uniform_(bias.data, 0.25, 0.5)

        # Initialize FC
        init.xavier_normal_(self.fc.weight.data)
        init.constant_(self.fc.bias.data, 0)

    def forward(self, inputs):
        # perform the Forward pass of the model
        output, (h_n, c_n) = self.lstm(inputs, self.init)
        output = output.reshape(-1, self.options['hidden_dim'])
        predictions = self.fc(output)
        predictions = predictions.reshape(-1,
                                          self.options['seq_len'],
                                          self.options['output_dim'])
        return predictions

        
class AudioToJoints(nn.Module):

    def __init__(self, options):
        super(AudioToJoints, self).__init__()

        # Instantiating the model
        self.init = None
        self.options = options

        # specify input features and hidden_dim
        self.lstm = nn.LSTM(self.options['input_dim'],
                            self.options['hidden_dim'],
                            batch_first=True).double()
        self.dropout = nn.Dropout(self.options['dropout']).double()
        self.fc = nn.Linear(self.options['hidden_dim'], self.options['output_dim']).double()

        self.initialize()

    def initialize(self):
        # Initialize LSTM Weights and Biases
        for layer in self.lstm._all_weights:
            for param_name in layer:
                if 'weight' in param_name:
                    weight = getattr(self.lstm, param_name)
                    init.xavier_normal_(weight.data)
                else:
                    bias = getattr(self.lstm, param_name)
                    init.uniform_(bias.data, 0.25, 0.5)

        # Initialize FC
        init.xavier_normal_(self.fc.weight.data)
        init.constant_(self.fc.bias.data, 0)

    def forward(self, inputs):
        # perform the Forward pass of the model
        output, (h_n, c_n) = self.lstm(inputs, self.init)
        # inputs.to(float)
        # output = output.view(-1, output.size()[-1])  # flatten before FC
        output = output.reshape(-1, self.options['hidden_dim'])
        dped_output = self.dropout(output)
        predictions = self.fc(dped_output)
        predictions = predictions.reshape(-1,
                                          self.options['seq_len'],
                                          self.options['output_dim'])
        return predictions

class AudioToJointsNonlinear(nn.Module):

    def __init__(self, options):
        super(AudioToJointsNonlinear, self).__init__()

        # Instantiating the model
        self.init = None
        self.options = options

        # specify input features and hidden_dim
        self.lstm = nn.LSTM(self.options['input_dim'],
                            self.options['hidden_dim'],
                            batch_first=True).double()
        self.dropout = nn.Dropout(self.options['dropout']).double()
        self.linear_one = nn.Linear(self.options['hidden_dim'], int(self.options['hidden_dim']/2))
        self.relu = nn.ReLU()
        self.fc = nn.Linear(int(self.options['hidden_dim']/2), self.options['output_dim']).double()
        self.initialize()

    def initialize(self):
        # Initialize LSTM Weights and Biases
        for layer in self.lstm._all_weights:
            for param_name in layer:
                if 'weight' in param_name:
                    weight = getattr(self.lstm, param_name)
                    init.xavier_normal_(weight.data)
                else:
                    bias = getattr(self.lstm, param_name)
                    init.uniform_(bias.data, 0.25, 0.5)

        # Initialize FC
        init.xavier_normal_(self.fc.weight.data)
        init.constant_(self.fc.bias.data, 0)

    def forward(self, inputs):
        # perform the Forward pass of the model
        output, (h_n, c_n) = self.lstm(inputs, self.init)
        # inputs.to(float)
        # output = output.view(-1, output.size()[-1])  # flatten before FC
        output = output.reshape(-1, self.options['hidden_dim'])
        dped_output = self.dropout(output)

        output = self.linear_one(dped_output)
        output = self.relu(output)
        predictions = self.fc(output)
        predictions = predictions.reshape(-1,
                                          self.options['seq_len'],
                                          self.options['output_dim'])
        return predictions


class AudioToJointsSeq2Seq(nn.Module):
    def __init__(self, options):
        super(AudioToJointsSeq2Seq, self).__init__()
        self.options = options

        # encoder 
        self.encoder = Encoder(self.options).double()
        self.decoder = Decoder(self.options).double()

        # decoder

    def forward(self, inputs, targets, teacher_forcing_ratio=0.5):
        # inputs:  [batch, seq_len, mfcc_features]
        # targets: [batch, seq_len, pose_features]
        batch_size = targets.shape[0]
        max_len = targets.shape[1]
        output_dim = self.options['output_dim']
        # container for decoder outputs
        outputs = torch.zeros(batch_size, max_len, output_dim)

        input, hidden_encode, cell_encode = self.encoder(inputs)
        
        for frame in range(1, max_len):
            output, hidden_decode, cell_decode = self.decoder(input, hidden_encode, cell_encode)
            outputs[:, frame, :] = output
            use_teacher_force = np.random.random() < teacher_forcing_ratio
            input = (targets[:, frame, :] if use_teacher_force else output)
        
        return outputs

class Encoder(nn.Module):
    def __init__(self, options):
        super(Encoder, self).__init__()
        self.options = options

        self.lstm = nn.LSTM(self.options['input_dim'],
                            self.options['hidden_dim'],
                            batch_first=True)
        self.linear_encode = nn.Linear(self.options['hidden_dim'],
                                       self.options['output_dim'])

    def forward(self, inputs):
        outputs, (hidden, cell) = self.lstm(inputs)

        output = outputs.reshape(-1, self.options['hidden_dim'])
        predictions = self.linear_encode(output)
        predictions = predictions.reshape(-1,
                                          self.options['seq_len'],
                                          self.options['output_dim'])
        predictions = predictions[:, 0, :]
        return predictions, hidden, cell

class Decoder(nn.Module):
    def __init__(self, options):
        super(Decoder, self).__init__()
        self.options = options

        self.lstm = nn.LSTM(self.options['output_dim'],
                            self.options['hidden_dim'],
                            batch_first=True)
        self.linear_decode = nn.Linear(self.options['hidden_dim'],
                                       self.options['output_dim'])
        self.dropout = nn.Dropout(self.options['dropout'])

    def forward(self, inputs, hidden, cell):
        # make 3 dimensional
        inputs = inputs.unsqueeze(1)
        output, (hidden, cell) = self.lstm(inputs, (hidden, cell))
        predicted = self.linear_decode(output)
        predicted = predicted.reshape(-1, self.options['output_dim'])

        return predicted, hidden, cell

        
