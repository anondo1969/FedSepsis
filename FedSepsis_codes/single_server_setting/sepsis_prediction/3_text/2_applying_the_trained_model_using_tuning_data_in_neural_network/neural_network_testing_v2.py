# -*- coding: utf-8 -*-
'''
@author: Mahbub Ul Alam (mahbub@dsv.su.se)
@version: 1.0+
@copyright: Copyright (c) Mahbub Ul Alam (mahbub@dsv.su.se)
@license : MIT License
'''

import features_extraction
from torch import LongTensor, FloatTensor
import torch
import pickle
import neural_network_architecture

def testing_neural_network_model(batch_size, config, data_from_test_file, model, exp_no, all_input_seq_list, all_output_seq_list, all_max_seq_length, all_batch_wise_ep_ids, time_window, change, time_stamp):
    
    '''
    Purpose:
        Testing the neural network model and saving the results

    Args:
        config : configuration file
        data_from_test_file : a python dictionary containing all the important information,
            
                        {
                        'episode_id_list' : the unique episode ids as a list,
                        'sepsis_patient_label_list' : patient label (1 or 0) on each episode id, 
                        'class_labels_name' : name of the unique labels (that is, name of the classes), 
                        'class_label_values' : actual categorical values of unique classes (starting from 1), 
                        'num_classes' : total number of classes, 
                        'total_data' : all the needed columns in panadas data frame format,
                        'input_size' : input feature shape for the neural network
                        }
                        
        model : the object of RNN_LSTM class, See 'neural_network_architecture.py' for the class details
            
    Returns:
        decoding_details: a python dictionary containg test results related data,
        
                        { 
                        'decoded_outputs': probability scores for each classes
                        'test_labels' : all the true labels
                        'episode_id_list_sorted' : episode ids sorted from maximum time steps to minimum
                        'seq_lengths_sorted' : sorted seq_lengths (time stamps) on each episode ids,
                        }
            
    '''
    
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    model.eval()
    
    save_dir = config.get('directories', 'exp_dir')
    episode_id_list=data_from_test_file['episode_id_list']
        
    whole_data=data_from_test_file['total_data']
    label_column=config.get('general', 'label_column')
    pad_integer = int(config.get('general', 'pad_integer'))
    input_size = data_from_test_file['input_size']
    num_classes = data_from_test_file['num_classes']+1 # for the dummy class in padding (0)
        
    ep_id_list=data_from_test_file['episode_id_list']
    #batch_size = 500#int(config.get('RNN_LSTM_baseline', 'batch_size')) # in a way so that all the test data is covered
        
    #all_input_seq_list, all_output_seq_list, all_output_string_seq_dict, all_max_seq_length, all_batch_wise_ep_ids = features_extraction.load_test_batch_data(ep_id_list, whole_data, label_column, batch_size=batch_size)
    #'''
    Y_true = []
    Y_predicted = []
    Y_predicted_score = []
    Episode_id_list_sorted = []
    Seq_lengths_sorted = []
        
        
    c=0
    for index in range(len(all_max_seq_length)):
            
        X_list = all_input_seq_list[index]
        Y_list = all_output_seq_list[index]
           
        max_seq_length = all_max_seq_length[index]
        batch_wise_episode_id_list = all_batch_wise_ep_ids[index]
            
        if len(batch_wise_episode_id_list) != len(Y_list):
            print('False' )
        
    
        #min_seq_length = min(list(map(len, Y_list)))
            
        #print('min_seq_length: '+str(min_seq_length))
        #print('max_seq_length: '+str(max_seq_length))
        
        seq_lengths = LongTensor(list(map(len, X_list))).to(device)
        #print(seq_lengths[5])
        
        seq_tensor = torch.Tensor(len(X_list), max_seq_length, input_size).float().to(device)
        seq_tensor = seq_tensor.fill_(pad_integer)
        
        for idx, (seq, seqlen) in enumerate(zip(X_list, seq_lengths)):
            #seq_tensor[idx, :seqlen] = FloatTensor(seq)
            seq_tensor[idx, :seqlen] = seq
    
        #print('seq_tensor.shape: '+str(seq_tensor.shape))
        
        
        label_tensor = torch.Tensor(len(Y_list), max_seq_length).long().to(device)
        label_tensor = label_tensor.fill_(pad_integer)
        for idx, (seq, seqlen) in enumerate(zip(Y_list, seq_lengths)):
            #label_tensor[idx, :seqlen] = LongTensor(seq)
            label_tensor[idx, :seqlen] = seq

        #print('label_tensor.shape: '+str(label_tensor.shape))
        
        
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]
        label_tensor = label_tensor[perm_idx]
        episode_id_list_sorted = batch_wise_episode_id_list[perm_idx.cpu()]                    
    
        outputs = model(seq_tensor, seq_lengths)
    
  
            
        outputs = torch.exp(outputs)
            
            
            
        outputs = outputs.view(-1, num_classes)
            
        #print(outputs)

        #print('final_outputs.shape: '+str(outputs.shape))

        labels = label_tensor.view(-1)
            

        #print('labels.shape: '+str(labels.shape))
            
        #append all the data here
        #----------------------------------------------
  
        y_true = labels.to(device).tolist()
          
        predicted_prob, y_predicted = torch.max(outputs.data, 1)
            
        #print('predicted_score.shape: '+str(predicted_score.shape))
        #print('y_predicted.shape: '+str(y_predicted.shape))
    
        y_predicted = y_predicted.to(device).tolist()
        predicted_prob = predicted_prob.to(device).tolist()
            
        predicted_score = []
        outputs_list = outputs.to(device).tolist()
            
        for timestep in range(len(outputs_list)):
            if sum(outputs_list[timestep]) != 0:
                predicted_score.append((outputs_list[timestep][2])/sum(outputs_list[timestep]))
            else:
                print('predicted score is zero')
                predicted_score.append(0)
                
        #print(predicted_score)
            
        #In Python 3.5 or greater:

        #z = {**x, **y}
    
        #print(str(len(Y_list))+':'+str(len(y_true)))
        c+=len(y_true)
            
        for value_index in range(len(y_true)):
                
            Y_true.append(y_true[value_index])
            Y_predicted.append(y_predicted[value_index])
            Y_predicted_score.append(predicted_score[value_index])
                
            
        seq_lengths = seq_lengths.to(device).tolist()
            
            
        for value_index in range(len(seq_lengths)):
                
            Episode_id_list_sorted.append(episode_id_list_sorted[value_index])
            Seq_lengths_sorted.append(seq_lengths[value_index])
                
             
        #print('Batch '+str(index+1)+' is completed')
            

    decoding_details = { 'Y_true': Y_true,
                         'Y_predicted' : Y_predicted,
                         'Y_predicted_score' : Y_predicted_score,
                         'episode_id_list_sorted' : Episode_id_list_sorted,
                         'seq_lengths_sorted' : Seq_lengths_sorted
                        }
    '''
    print ('Total Y_true: ' + str(len(Y_true)))
    print ('Total Y_predicted: ' + str(len(Y_predicted)))
    print ('Total episode_id_list_sorted: ' + str(len(Episode_id_list_sorted)))
    print ('Total Seq_lengths_sorted: ' + str(len(Seq_lengths_sorted)))
    print(c)
    '''
        
    #with open(save_dir+str(time_window)+'_hours_'+imputation+'_'+change+"_decoding_details_"+str(exp_no), "wb") as fp:
        #pickle.dump(decoding_details, fp)
        
    with open(save_dir+str(time_window)+'_hours_'+change+"_"+time_stamp+"_decoding_details_"+str(exp_no), "wb") as fp:
        pickle.dump(decoding_details, fp)
            
    print('Tune data applied successfully in the trained model')
           
        
                
    #----------------------------------------------
    '''
    print ('\nNeural networks testing starts.\n------------------------------------------------------------------\n')
            
    print('min_seq_length: '+str(min_seq_length))
    print('max_seq_length: '+str(max_seq_length))
    print('seq_tensor.shape: '+str(seq_tensor.shape))
    print('label_tensor.shape: '+str(label_tensor.shape))
    print('final_outputs.shape: '+str(outputs.shape))
    print('labels.shape: '+str(labels.shape))
        
    print ('\nNeural networks testing ends.\n------------------------------------------------------------------\n')
    '''
        
    return decoding_details
    
    

def load_trained_model(config, hyper_parameters, data_from_tune_file, experiment_number, time_window, change, time_stamp):
    
    '''
    Purpose:
        Retrieving the model of the trained neural networks from the experiment directory

    Args:
        config : configuration file
        data_from_train_file : a python dictionary containing all the important information,
            
                        {
                        'episode_id_list' : the unique episode ids as a list,
                        'sepsis_patient_label_list' : patient label (1 or 0) on each episode id, 
                        'class_labels_name' : name of the unique labels (that is, name of the classes), 
                        'class_label_values' : actual categorical values of unique classes (starting from 1), 
                        'num_classes' : total number of classes, 
                        'total_data' : all the needed columns in panadas data frame format,
                        'input_size' : input feature shape for the neural network
                        }
            
    Returns:
        model: the object of RNN_GRU class, See 'neural_network_architecture.py' for the class details
            
    ''' 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    save_dir = config.get('directories', 'exp_dir')
    
    #model_name = str(time_window)+'_hours_'+imputation+'_'+change+'_trained_model'
    model_name = str(time_window)+'_hours_'+change+'_trained_model_'+time_stamp
    
    hidden_size = int(hyper_parameters[3])
    num_layers = int(hyper_parameters[4])
    num_classes = data_from_tune_file['num_classes']+1 # additional one is for the dummy class in padding (0)
    #batch_size = int(config.get('hyper_parameter_tuning', 'batch_size'))
    #num_epochs = int(hyper_parameters[6])#int(config.get('RNN_LSTM', 'num_epochs'))
    #learning_rate = hyper_parameters[0]
    input_size = data_from_tune_file['input_size']
    #beta_one = hyper_parameters[1]
    #beta_two =hyper_parameters[2] #
    dropout = hyper_parameters[5]
    
    if num_layers==1: dropout=0
    
    # Load the model from Saved file
    model = neural_network_architecture.RNN_LSTM(input_size, hidden_size, num_layers, num_classes, dropout).to(device)
    
    try:
        model.load_state_dict(torch.load(save_dir+model_name+'_'+str(experiment_number)+'.ckpt', map_location='cuda' if torch.cuda.is_available() else 'cpu'))
        #model.eval()
        #print('\nTrained model loaded successfully\n------------------------------------------------------------------')
        print('Trained model loaded successfully')
        return model
        
    except FileNotFoundError:
        print('could not find the model, please check again')
    
