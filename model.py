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
