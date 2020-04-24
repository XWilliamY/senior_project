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

class AudioToJoints(nn.Module):

    def __init__(self, options):
        super(AudioToJoints, self).__init__()

        # Instantiating the model
        self.init = None
        self.options = options

        # specify input features and hidden_dim
        self.lstm = nn.LSTM(self.options['input_dim'],
                            self.options['hidden_dim'])
        self.dropout = nn.Dropout(self.options['dropout'])
        self.fc = nn.Linear(self.options['hidden_dim'], self.options['output_dim'])

        self.initialize()

    def initialize(self):
        # Initialize LSTM Weights and Biases
        for layer in self.lstm._all_weights:
            for param_name in layer:
                if 'weight' in param_name:
                    weight = getattr(self.lstm, param_name)
                    init.xavier_normal_(weight.data)
                    print(weight.data.dtype)
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
        output = output.view(-1, output.size()[-1])  # flatten before FC
        dped_output = self.dropout(output)
        predictions = self.fc(dped_output)
        return predictions
