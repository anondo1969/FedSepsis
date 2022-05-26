# -*- coding: utf-8 -*-

'''
@author: Mahbub Ul Alam (mahbub@dsv.su.se)
@date: 25.01.2019
@version: 1.0+
@copyright: Copyright (c)  2019-2020, Mahbub Ul Alam (mahbub@dsv.su.se)
@license : MIT License
'''

import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

class RNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, seq_tensor, seq_lengths):
        
        
        
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, seq_tensor.size(0), self.hidden_size).to(self.device) 
        c0 = torch.zeros(self.num_layers, seq_tensor.size(0), self.hidden_size).to(self.device)
        
        #pack_padded_sequence and pad_packed_sequence are used to pad the variable length time step data
        #per episode in a fixed length one, see their documentation in pytorch for details
        #packed_input = pack_padded_sequence(seq_tensor, seq_lengths, batch_first=True)
        
        packed_input = pack_padded_sequence(seq_tensor, seq_lengths.cpu(), batch_first=True)
        
        #print('packed_input.data.shape: '+str(packed_input.data.shape))
        
        
        # Forward propagate LSTM
        packed_output, (ht, ct) = self.lstm(packed_input, (h0, c0))
        
        
        #print('packed_output.data.shape: '+str(packed_output.data.shape))
        
        
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)

        
        #print('output.shape: '+str(output.shape))
        
        # Decode the hidden state of the last time step
        #out = self.fc(out[:, -1, :])
        #output = self.relu(output)
        #output = self.relu(output)
        #output = self.tanh(output)
        
        out = self.relu(output)
        out = self.fc(out) #Final Output
        
        
        
        #print(out.shape)
        out = torch.nn.functional.log_softmax(out, dim=1)
        
        
        return out
        
def hyper_parameter_tuner(config, total_experiments):
    
    np.random.seed(1969)
    
    #total_experiments = int(config.get('hyper_parameter_tuning', 'total_experiments'))
    
    alpha = int(config.get('hyper_parameter_tuning', 'alpha'))
    
    alpha_values = alpha*np.random.rand(total_experiments)
    
    alpha_values = np.power(10, alpha_values)
    
    beta_one = int(config.get('hyper_parameter_tuning', 'beta_one'))
    
    beta_one_values = beta_one*np.random.rand(total_experiments)
    
    beta_one_values = 1 - np.power(10, beta_one_values)
    
    beta_two = int(config.get('hyper_parameter_tuning', 'beta_two'))
    
    beta_two_values = beta_two*np.random.rand(total_experiments)
    
    beta_two_values = 1 - np.power(10, beta_two_values)
    
    hidden_units = config.get('hyper_parameter_tuning', 'hidden_units').split()
    
    hidden_units = np.array([int(value) for value in hidden_units])
    
    hidden_units_values = hidden_units[np.random.randint(0,len(hidden_units),total_experiments)]
    
    hidden_layers = config.get('hyper_parameter_tuning', 'hidden_layers').split()
    
    hidden_layers = np.array([int(value) for value in hidden_layers])
    
    hidden_layers_values = hidden_layers[np.random.randint(0,len(hidden_layers),total_experiments)]
    
    drop_out = config.get('hyper_parameter_tuning', 'drop_out').split()
    
    drop_out = np.array([int(value) for value in drop_out])
    
    drop_out_values = drop_out[np.random.randint(0,len(drop_out),total_experiments)] / 100.0
    
    epochs = config.get('hyper_parameter_tuning', 'epochs').split()
    
    epochs = np.array([int(value) for value in epochs])
    
    epochs_values = epochs[np.random.randint(0,len(epochs),total_experiments)]
    
    
    all_hyper_parameters = []
    for experiment_number in range(total_experiments):
        all_hyper_parameters.append([alpha_values[experiment_number], #0
                                 beta_one_values[experiment_number], #1
                                 beta_two_values[experiment_number], #2
                                 hidden_units_values[experiment_number], #3
                                 hidden_layers_values[experiment_number], #4
                                 drop_out_values[experiment_number], #5
                                 epochs_values[experiment_number]] ) #6

    return all_hyper_parameters

