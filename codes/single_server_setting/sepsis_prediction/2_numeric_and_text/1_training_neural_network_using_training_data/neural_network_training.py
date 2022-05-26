# -*- coding: utf-8 -*-
'''
@author: Mahbub Ul Alam (mahbub@dsv.su.se)
@date: 25.01.2019
@version: 1.0+
@copyright: Copyright (c)  2019-2020, Mahbub Ul Alam (mahbub@dsv.su.se)
@license : MIT License
'''

import neural_network_architecture
import features_extraction
import torch.nn as nn
import torch
from torch import LongTensor, FloatTensor
import numpy as np


def training_neural_network_model(batch_size, config, data_from_train_file, hyper_parameters, experiment_number, time_window, imputation, all_pos_all_neg, change, time_stamp):
    
    '''
    Purpose:
        Training the neural network model and saving the results

    Args:
        config : configuration file
        data_from_train_file : a python dictionary containing all the important information,
            
                        {
                        'episode_id_list' : the unique episode ids as a list,
                        'sepsis_patient_label_list' : patient label (1 or 0) on each episode id, 
                        'class_label_values' : actual categorical values of unique classes (starting from 1), 
                        'num_classes' : total number of classes, 
                        'total_data' : all the needed columns in panadas data frame format,
                        'input_size' : input feature shape for the neural network
                        }
            
    Returns:
        model: the object of RNN_GRU class, See 'neural_network_architecture.py' for the class details
            
    '''
    #print('alpha\tbeta_one\tbeta_two\thidden_units\thidden_layers\tdrop_out\tepochs')
    #print(hyper_parameters)
    #print()
    
    save_dir = config.get('directories', 'exp_dir')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    hidden_size = int(hyper_parameters[3])
    num_layers = int(hyper_parameters[4])
    num_classes = data_from_train_file['num_classes']+1 # additional one is for the dummy class in padding (0)
    #batch_size = int(config.get('hyper_parameter_tuning', 'batch_size'))
    num_epochs = int(hyper_parameters[6])#int(config.get('RNN_LSTM', 'num_epochs'))
    learning_rate = hyper_parameters[0]
    input_size = data_from_train_file['input_size']
    beta_one = hyper_parameters[1]
    beta_two =hyper_parameters[2] #
    dropout = hyper_parameters[5] 

    data=data_from_train_file['episode_id_list']
    target=data_from_train_file['sepsis_patient_label_list']
    whole_data=data_from_train_file['total_data']
    row_df = data_from_train_file['row_df']
    embedding_dict = data_from_train_file['embedding_dict']
    embedding_size = data_from_train_file['embedding_size']
    label_column=config.get('general', 'label_column')
    total_episodes = len(data)
    
    pad_integer = int(config.get('general', 'pad_integer'))
    
    model = neural_network_architecture.RNN_LSTM(input_size, hidden_size, num_layers, num_classes, dropout).to(device)
    
    criterion = nn.CrossEntropyLoss(size_average=True, ignore_index=pad_integer)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(beta_one, beta_two))
    #optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    #optimizer = torch.optim.SparseAdam(model.parameters(), lr=learning_rate)
    
    train_loader = features_extraction.create_train_data_upsampled_numbers(data, target, batch_size)
    #loss_info_file_name = save_dir+'training_loss_information_'+str(time_window)+'_hours_'+imputation+'.txt'
    loss_info_file_name = save_dir+'training_loss_information_'+str(time_window)+'_hours_'+imputation+'_'+time_stamp+'.txt'
    loss_information_file = open(loss_info_file_name,'a')
    loss_information_file.write('\n')
    loss_information_file.write('Experiment number : '+str(experiment_number)+'\n')
    loss_information_file.close()
    
    for epoch in range(num_epochs):

        for batch_count, (batch_wise_episode_id_list, target) in enumerate(train_loader):
            
            
            #print ("batch index {}, 0/1: {}/{}".format(
                    #batch_count,
                    #len(np.where(target.numpy() == 0)[0]),
                    #len(np.where(target.numpy() == 1)[0])))
            
    
            X_list, Y_list, max_seq_length = features_extraction.load_train_batch_data(batch_wise_episode_id_list, whole_data, row_df, embedding_dict, embedding_size, label_column, all_pos_all_neg)
    
    
            #print('max_seq_length: '+str(max_seq_length))
        
            seq_lengths = LongTensor(list(map(len, X_list))).to(device)
            #print('seq_lengths: '+ str(seq_lengths))
        
            seq_tensor = torch.Tensor(batch_size, max_seq_length, input_size).float().to(device)
            seq_tensor = seq_tensor.fill_(pad_integer)
        
            for idx, (seq, seqlen) in enumerate(zip(X_list, seq_lengths)):
                #print (seq)
                seq_tensor[idx, :seqlen] = FloatTensor(seq)
    
            #print('seq_tensor.shape: '+str(seq_tensor.shape))
        
        
            label_tensor = torch.Tensor(batch_size, max_seq_length).long().to(device)
            label_tensor = label_tensor.fill_(pad_integer)
            for idx, (seq, seqlen) in enumerate(zip(Y_list, seq_lengths)):
                label_tensor[idx, :seqlen] = LongTensor(seq)

            #print('label_tensor.shape: '+str(label_tensor.shape))
        
        
            seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
            seq_tensor = seq_tensor[perm_idx]
            label_tensor = label_tensor[perm_idx]

                      
            # Forward pass
            outputs = model(seq_tensor, seq_lengths)
        
            outputs = outputs.view(-1, num_classes)

            #print('final_outputs.shape: '+str(outputs.shape))

            labels = label_tensor.view(-1)

            #print('labels.shape: '+str(labels.shape))

        
            loss = criterion(outputs, labels)
        
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                       
            #'''
            if (batch_count+1) % 20 == 0:
                
                loss_information_file = open(loss_info_file_name,'a')
                loss_information_file.write('Epoch [{}/{}], Episodes [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, num_epochs, (batch_count+1)*batch_size, total_episodes, loss.item()))
                loss_information_file.write('\n')
                loss_information_file.close()
                
                #print ('Epoch [{}/{}], Episodes [{}/{}], Loss: {:.4f}' 
                       #.format(epoch+1, num_epochs, (batch_count+1)*batch_size, total_episodes, loss.item()))
            #'''
 
    # Save the model checkpoint
    #model_name = str(time_window)+'_hours_'+imputation+'_'+change+'_trained_model'
    model_name = str(time_window)+'_hours_'+imputation+'_'+change+'_trained_model_'+time_stamp
    torch.save(model.state_dict(), save_dir+model_name+'_'+str(experiment_number)+'.ckpt')
    
    return model
