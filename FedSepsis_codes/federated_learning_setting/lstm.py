# -*- coding: utf-8 -*-
'''
@author: Mahbub Ul Alam (mahbub@dsv.su.se)
@version: 1.0+
@copyright: Copyright (c) Mahbub Ul Alam (mahbub@dsv.su.se)
@license : MIT License
'''

import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, seq_tensor, seq_lengths):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, seq_tensor.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, seq_tensor.size(0), self.hidden_size).to(self.device)

        # pack_padded_sequence and pad_packed_sequence are used to pad the variable length time step data
        # per episode in a fixed length one, see their documentation in pytorch for details
        # packed_input = pack_padded_sequence(seq_tensor, seq_lengths, batch_first=True)

        packed_input = pack_padded_sequence(seq_tensor, seq_lengths.cpu(), batch_first=True)

        # print('packed_input.data.shape: '+str(packed_input.data.shape))

        # Forward propagate LSTM
        packed_output, (ht, ct) = self.lstm(packed_input, (h0, c0))

        # print('packed_output.data.shape: '+str(packed_output.data.shape))

        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)

        # print('output.shape: '+str(output.shape))

        # Decode the hidden state of the last time step
        # out = self.fc(out[:, -1, :])
        out = self.relu(output)
        out = self.fc(out)
        # print(out.shape)
        out = torch.nn.functional.log_softmax(out, dim=1)

        return out
