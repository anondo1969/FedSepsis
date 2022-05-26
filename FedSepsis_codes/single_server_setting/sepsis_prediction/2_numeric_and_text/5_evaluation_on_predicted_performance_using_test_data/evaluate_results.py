# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
'''
@author: Mahbub Ul Alam (mahbub@dsv.su.se)
@date: 25.01.2019
@version: 1.0+
@copyright: Copyright (c)  2019-2020, Mahbub Ul Alam (mahbub@dsv.su.se)
@license : MIT License
'''
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score, roc_curve, auc
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta


def show_error(save_dir, file_name):
    
    print('\nFile or Directory Not Found Error:\n\nDid not find the '+file_name+' in the location '+save_dir+'\n\nMake sure of these followings, \n1. you have the right permission to create the directory in that location\n2. you have enabled your encrypted directory to access it.\n3. The experiment directory '+save_dir+' is the correct one\n4. you have executed the previous module(s) successfully before at least once.\n')
         
def load_decoding_details(config, value, time_window, imputation, change, time_stamp):
    
    '''
    Purpose:
        Retrieving the results of testing the neural networks from the experiment directory

    Args:
        config : configuration file
            
    Returns:
        decoding_details: a python dictionary containg test results related data,
        
                        { 
                        'decoded_outputs': probability scores for each classes
                        'test_labels' : all the true labels
                        'episode_id_list_sorted' : episode ids sorted from maximum time steps to minimum
                        'seq_lengths_sorted' : sorted seq_lengths (time stamps) on each episode ids,
                        'output_string_seq_dict' : label names for the classes in a dictionary where key is the episode id
                        }
            
    ''' 
    
    save_dir = config.get('directories', 'exp_dir')

    
    try:
        #with open(save_dir+str(time_window)+'_hours_'+imputation+'_'+change+"_decoding_details_"+str(value), "rb") as fp:
            #decoding_details = pickle.load(fp)
            
        with open(save_dir+str(time_window)+'_hours_'+imputation+'_'+change+"_"+time_stamp+"_test_decoding_details_"+str(value), "rb") as fp:
            decoding_details = pickle.load(fp)
            
        #print('decoding_details loaded')
    
        return decoding_details
    
    except FileNotFoundError:
        show_error(save_dir, 'decoding_details file')


def remove_dummy_pad_value_zero_from_timestep_wise_y_true_and_y_predict(decoding_details, config):
    
    Y = decoding_details['Y_true']
    
    pad_integer = int(config.get('general', 'pad_integer'))
    
    Y = np.array(Y)
    
    Y_with_dummy_class = Y
       
    Y_predicted = decoding_details['Y_predicted']
    
    Y_predicted_with_dummy_class = Y_predicted
    
    Y_predicted = np.array(Y_predicted)
    
    Y_predicted_score = decoding_details['Y_predicted_score']
    
    Y_predicted_score_with_dummy_class = Y_predicted_score
    
    Y_predicted_score = np.array(Y_predicted_score)
    
    # locate the indexes where label is pad value(0)
    pad_index = np.where(Y==pad_integer)
    
    # make 0 integer a numpy array
    pad_array = np.array([pad_integer])
    
    # remove all the pad (0) values
    Y = Y[~np.in1d(Y,pad_array)]
    
    # remove all the values same index with pad (0) values
    Y_predicted = np.delete(Y_predicted, pad_index)
    Y_predicted_score = np.delete(Y_predicted_score, pad_index)
    
    #making all the dummy value predictions as negative
    count = 0
    for value_index in range(len(Y_predicted)):
        if Y_predicted[value_index]==0:
            count+=1
            Y_predicted[value_index] = 1
            
            
    #print('Total dummy predictions: '+str(count))

    timestep_wise_y_true_and_y_predicts = { 'Y': Y, #Y_true
                  'Y_predicted' : Y_predicted,
                  'Y_predicted_score' : Y_predicted_score,
                  'Y_with_dummy_class' : Y_with_dummy_class,
                  'Y_predicted_with_dummy_class' : Y_predicted_with_dummy_class,
                  'Y_predicted_score_with_dummy_class' : Y_predicted_score_with_dummy_class
                 }
                
    return timestep_wise_y_true_and_y_predicts
    
def compute_episodes_wise_y_true_and_y_predicts(timestep_wise_y_true_and_y_predicts, decoding_details, config, settings, range_limit, time_window):
    

    timestep_wise_Y_true = timestep_wise_y_true_and_y_predicts['Y']
    timestep_wise_Y_predicted = timestep_wise_y_true_and_y_predicts['Y_predicted']
    timestep_wise_Y_predicted_score = timestep_wise_y_true_and_y_predicts['Y_predicted_score']
    
    seq_lengths = decoding_details['seq_lengths_sorted']
    ep_ids = np.array(decoding_details['episode_id_list_sorted'])
    #string_seq_dict = decoding_details['output_string_seq_dict']
    
    ep_id_to_y_true_sequence = {}
    ep_id_to_y_pred_sequence = {}
    ep_id_to_y_pred_pos_prob_score_sequence = {}
    ep_id_to_y_pred = {}
    ep_id_to_y_true = {}
    
    episode_id_count=0
    starting_index = 0
    starting_index = 0
    
    Y_true_episode = []
    Y_pred_episode = []
    Y_pred_pos_prob_score_episode = []
    
    sepsis_index_time_dict, sepsis_onset_time_dict = load_time_dicts(config, time_window)
    
    for length in seq_lengths:
        
        last_index = starting_index+length-1
        
        y_true_sequence = timestep_wise_Y_true[starting_index:last_index+1]
        y_pred_sequence = timestep_wise_Y_predicted[starting_index:last_index+1]
        y_pred_pos_prob_score_sequence = timestep_wise_Y_predicted_score[starting_index:last_index+1]
        
        ep_id = ep_ids[episode_id_count]
        
        ep_id_to_y_true_sequence[ep_id] = y_true_sequence
        ep_id_to_y_pred_sequence[ep_id] = y_pred_sequence
        ep_id_to_y_pred_pos_prob_score_sequence[ep_id] = y_pred_sequence
        
        Y_true_value = y_true_sequence[-1]
            
        if settings == 3:
            
            if range_limit!=0 and len(y_pred_sequence) > ((range_limit / time_window)+2):
                y_pred_sequence = y_pred_sequence[-int((range_limit / time_window)+2):]
            
        if 2 in y_pred_sequence:
            
            Y_pred_episode.append(2)
            Y_pred_pos_prob_score_episode.append(y_pred_pos_prob_score_sequence[y_pred_sequence.tolist().index(2)])
            ep_id_to_y_pred[ep_id] = 2
            
            if settings == 1 and Y_true_value==2 and calculate_value_exceeded_time_limit(config, y_pred_sequence, sepsis_index_time_dict, sepsis_onset_time_dict, ep_id, range_limit)==True:
                # we have to make it false positive if TP is more than time limit, True means exceeded, False means it is inside time limit
                Y_true_value = 1
        else:
            Y_pred_episode.append(1)
            Y_pred_pos_prob_score_episode.append(y_pred_pos_prob_score_sequence[-1])
            ep_id_to_y_pred[ep_id] = 1
        
        Y_true_episode.append(Y_true_value)
        ep_id_to_y_true[ep_id] = Y_true_value
        
        episode_id_count+=1
        starting_index = starting_index+length
        
    
        
    episodes_based_y_true_and_y_predicts = { 'Y_true_episode' : Y_true_episode,
                  'Y_pred_episode' : Y_pred_episode,
                  'Y_pred_pos_prob_score_episode' : Y_pred_pos_prob_score_episode,
                  'ep_id_to_y_true_sequence' : ep_id_to_y_true_sequence,
                  'ep_id_to_y_pred_sequence' : ep_id_to_y_pred_sequence,
                  'ep_id_to_y_pred_pos_prob_score_sequence' : ep_id_to_y_pred_pos_prob_score_sequence,
                  'ep_id_to_y_pred' : ep_id_to_y_pred,
                  'ep_id_to_y_true' : ep_id_to_y_true
                 }
    return episodes_based_y_true_and_y_predicts
    

                 
def compute_episode_wise_evaluation_scores(episode_based_y_true_and_y_predicts, config):
    
    
    Y_true = episode_based_y_true_and_y_predicts['Y_true_episode']
    
    
    Y_pred = episode_based_y_true_and_y_predicts['Y_pred_episode']
    Y_pred_score = episode_based_y_true_and_y_predicts['Y_pred_pos_prob_score_episode']
          
    evaluation_scores = compute_evaluation_score(Y_true, Y_pred, Y_pred_score)

    return evaluation_scores
    
def calculate_value_exceeded_time_limit(config, y_pred_sequence, sepsis_index_time_dict, sepsis_onset_time_dict, ep_id, range_limit):
    
    
    if range_limit==0:
        return False
        
    
    fmt = '%Y-%m-%d %H:%M:%S'

    t = timedelta(hours = range_limit)
    
    first_predict_index = y_pred_sequence.tolist().index(2)
                
    if len(y_pred_sequence)==1 or first_predict_index== len(y_pred_sequence)-1:
        prediction_time = sepsis_index_time_dict[ep_id][first_predict_index]
        
    else:
        prediction_time = sepsis_index_time_dict[ep_id][first_predict_index+1]
       
                                       
    prediction_time = datetime.strptime(str(prediction_time), fmt)
    
    
    onset_time = sepsis_onset_time_dict[ep_id]
    onset_time = datetime.strptime(str(onset_time), fmt)
    
                
    if onset_time-prediction_time > t:
        return True
    else:
        return False

def compute_episodes_wise_earliness(episodes_based_y_true_and_y_predicts, config, settings, range_limit, time_window):
    
    
    range_limit_hour = timedelta(hours = range_limit)
    
    sepsis_index_time_dict, sepsis_onset_time_dict = load_time_dicts(config, time_window)
    
    
    fmt = '%Y-%m-%d %H:%M:%S'

    t = timedelta(hours = 0)
    
    count = 0
    
    ep_id_to_y_pred = episodes_based_y_true_and_y_predicts['ep_id_to_y_pred']
    ep_id_to_y_true = episodes_based_y_true_and_y_predicts['ep_id_to_y_true']
    ep_id_to_y_pred_sequence = episodes_based_y_true_and_y_predicts['ep_id_to_y_pred_sequence']
    
    for ep_id in ep_id_to_y_pred:
        
        
        if ep_id_to_y_pred[ep_id] == 2 and ep_id_to_y_true[ep_id] == 2:
            
            '''
            print(ep_id_to_y_pred_sequence[ep_id])
            print(episodes_based_y_true_and_y_predicts['ep_id_to_y_true_sequence'][ep_id])
            print(sepsis_onset_time_dict[ep_id])
            '''
            
            count+=1
   
            y_pred_sequence = ep_id_to_y_pred_sequence[ep_id]
            
            first_predict_index = y_pred_sequence.tolist().index(2)
                        
            if len(y_pred_sequence)==1 or first_predict_index== len(y_pred_sequence)-1:
                prediction_time = sepsis_index_time_dict[ep_id][first_predict_index]
                zero_index_time = sepsis_index_time_dict[ep_id][-1]
            else:
                prediction_time = sepsis_index_time_dict[ep_id][first_predict_index+1]
                zero_index_time = sepsis_index_time_dict[ep_id][-2]
                    
            onset_time = sepsis_onset_time_dict[ep_id]
                        
                                        
            prediction_time = datetime.strptime(str(prediction_time), fmt)
            onset_time = datetime.strptime(str(onset_time), fmt)
            zero_index_time = datetime.strptime(str(zero_index_time), fmt)
                        
            if settings == 2 and range_limit!=0 and onset_time-prediction_time>range_limit_hour:
                t += range_limit_hour
            else:        
                t += (onset_time-prediction_time)
            

    if count!=0:
        
        average_prior_prediction_in_hour = t/float(count)
        average_prior_prediction_in_hour = average_prior_prediction_in_hour.total_seconds() / float(3600)
    else:
        average_prior_prediction_in_hour = 0
  
    return average_prior_prediction_in_hour



def save_sepsis_time_info(config, time_window):
    
    
    data_dir = config.get('directories', 'data_dir')
    
    type_data = config.get('general', 'type')
    
    
    
    df = pd.read_csv(data_dir+type_data+'_data_'+str(time_window)+'_hours.csv')
   

    episode_id_list = df['icustay_id'].unique().tolist()

    sepsis_onset_time_dict = dict()
    sepsis_index_time_dict = dict()
    
    for ep_id in episode_id_list:
        
        seq = df.loc[df['icustay_id'] == ep_id]
        
        #sepsis_index_time = seq['end_time'].tolist()
        
        sepsis_index_time = pd.to_datetime(seq['end_time']).tolist()
        
        sepsis_index_time_dict[ep_id]= sepsis_index_time
    
        sepsis_onset_time_dict[ep_id]= sepsis_index_time[-1]
            
    with open(data_dir+'sepsis_onset_time_dict_'+type_data+'_'+str(time_window)+'_hours', "wb") as fp:
        pickle.dump(sepsis_onset_time_dict, fp)
        
    print(data_dir+'sepsis_onset_time_dict is saved')
    
    with open(data_dir+'sepsis_index_time_dict_'+type_data+'_'+str(time_window)+'_hours', "wb") as fp:
        pickle.dump(sepsis_index_time_dict, fp)


    print(data_dir+'sepsis_index_time_dict is saved')
    
    return sepsis_onset_time_dict, sepsis_index_time_dict
        

    
def load_time_dicts(config, time_window):
    
    data_dir = config.get('directories', 'data_dir')
    
    type_data = config.get('general', 'type')
    
    
    #print (save_dir)
    
    try:
        with open(data_dir+'sepsis_index_time_dict_'+type_data+'_'+str(time_window)+'_hours', "rb") as fp:
            sepsis_index_time_dict = pickle.load(fp)
        
        with open(data_dir+'sepsis_onset_time_dict_'+type_data+'_'+str(time_window)+'_hours', "rb") as fp:
            sepsis_onset_time_dict = pickle.load(fp)
        #print('Files loaded')
        
    except:
        sepsis_onset_time_dict, sepsis_index_time_dict = save_sepsis_time_info(config, time_window)
        
    return sepsis_index_time_dict, sepsis_onset_time_dict
    
def compute_evaluation_score(Y_true, Y_pred, Y_pred_score):
    
    try:
        accuracy = accuracy_score(Y_true, Y_pred) * 100.0
    except:
        accuracy = np.nan
    
    Y = [value-1 for value in Y_true]
    Y_predicted = [value-1 for value in Y_pred]

    try:
        true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(Y, Y_predicted).ravel()
    except:
        true_negatives = np.nan
        false_positives = np.nan
        false_negatives = np.nan
        true_positives = np.nan
        
    try:
        recall = recall_score(Y, Y_predicted, average='binary') * 100.0
    except:
        recall = np.nan
        
    try:
        precision = precision_score(Y, Y_predicted, average='binary') * 100.0
    except:
        precision = np.nan
        
    try:
        F1_score = f1_score(Y, Y_predicted, average='binary') * 100.0
    except:
        F1_score = np.nan
    
    
    
        
    try:
        fpr, tpr, thresholds = roc_curve(Y, Y_pred_score)
    
        AU_ROC = auc(fpr, tpr) * 100.0

    except:
        print('Exception in fpr and tpr')
        AU_ROC = np.nan
    
    try:   
        AU_PRC = average_precision_score(Y, Y_pred_score) * 100.0
    except:
        print('Exception in average_precision_score')
        AU_PRC = np.nan
        
    evaluation_scores = [accuracy, 
                         true_positives, 
                         true_negatives, 
                         false_positives, 
                         false_negatives,
                         recall,
                         precision,
                         F1_score,
                         AU_ROC,
                         AU_PRC]
    
    return evaluation_scores
    
    

def write_experiment_information(config, average_prior_prediction_in_hour, episode_wise_evaluation_scores, settings, range_limit, value, time_window, imputation, change, header, clinicalBERT, embedding_type, batch_size, time_stamp):
    
    data_dir = config.get('directories', 'evaluation_dir')
    
  
    #experiment_information_file_name = config.get('general', 'information_file_name')
    experiment_information_file_name = config.get('general', 'information_file_name')+'_'+time_stamp+'.tsv'
    
    experiment_information_file = open(data_dir+experiment_information_file_name,'a')
    
    
    #if change == 'with_train_label_change' and time_window==8 and imputation=='imputed' and settings == 1 and range_limit == 0 and value==1:
        
    header_line = 'experiment_number\ttime_stamp\tbatch_size\tclinicalBERT\tembedding_type\ttime_window\ttraining_label_change\timputation\tsettings\trange_limit\tearliness\tepisode_wise_accuracy\tepisode_wise_true_positives\tepisode_wise_true_negatives\tepisode_wise_false_positives\tepisode_wise_false_negatives\tepisode_wise_recall\tepisode_wise_precision\tepisode_wise_f1_score\tepisode_wise_AU_ROC\tepisode_wise_AU_PRC'
    
    if header:
        experiment_information_file.write(header_line)
        
        #print(header_line)
        
    experiment_wise_information = []
    
    experiment_wise_information.append(str(value))
    
    experiment_wise_information.append(time_stamp)
    
    experiment_wise_information.append(str(batch_size))
    
    experiment_wise_information.append(clinicalBERT)
    
    experiment_wise_information.append(embedding_type)
        
    experiment_wise_information.append(str(time_window))
    
    experiment_wise_information.append(change)
    
    experiment_wise_information.append(imputation)
    
    experiment_wise_information.append(str(settings))
    
    experiment_wise_information.append(str(range_limit))
       
    experiment_wise_information.append(str(average_prior_prediction_in_hour))
         
    for episode_wise_evaluation_score in episode_wise_evaluation_scores:
        experiment_wise_information.append(str(episode_wise_evaluation_score))
            
    experiment_information_line = '\t'.join(experiment_wise_information)
        
    experiment_information_file.write('\n'+experiment_information_line)
        
    #print(experiment_information_line)
 
    experiment_information_file.close() 