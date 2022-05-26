'''
@author: Mahbub Ul Alam (mahbub@dsv.su.se)
@date: 25.01.2019
@version: 1.0+
@copyright: Copyright (c)  2019-2020, Mahbub Ul Alam (mahbub@dsv.su.se)
@license : MIT License
'''

'''@file early_sepsis_detector.py
run this file to go through the complete neural net training, testing and evaluating procedures, 
look at the 'early_sepsis_detector_configuration.cfg' file to modify the settings'''

from six.moves import configparser
import features_extraction
import neural_network_architecture, neural_network_testing_v2
import os
from datetime import datetime
import sys
import pandas as pd

start_time=datetime.now()

hidden_units= int(sys.argv[1])
hidden_layers = int(sys.argv[2])
drop_out = float(sys.argv[3])/100.00
epochs = int(sys.argv[4])
imputation=sys.argv[5]

experiment_number = int(sys.argv[6])
batch_size = int(sys.argv[7])
clinicalBERT = sys.argv[8]
embedding_type = sys.argv[9]
time_stamp = sys.argv[10]

#if all_pos_all_neg==1:
#    change = 'with_train_label_change'
#else:
#    change = 'without_train_label_change'

#HYPER_PARAMETERS_TUNING = True


#read config file
config = configparser.ConfigParser()
config.read('testing_configuration.cfg')


#create the directory if it does not exist
save_dir = config.get('directories', 'exp_dir')
data_dir = config.get('directories', 'data_dir')

all_hyper_parameters = neural_network_architecture.hyper_parameter_tuner(config, 50)

all_pos_all_neg=1
change = 'with_train_label_change'
time_window=1
hyper_parameters = all_hyper_parameters[experiment_number]

custom_parameters=[]

for value_index in range(3):
    custom_parameters.append(hyper_parameters[value_index])

custom_parameters.append(hidden_units)
custom_parameters.append(hidden_layers)
custom_parameters.append(drop_out)
custom_parameters.append(epochs)



        

data_from_test_file = features_extraction.load_data(time_window, imputation, config, 'test', clinicalBERT, embedding_type)

        

all_input_seq_list, all_output_seq_list, all_max_seq_length, batch_wise_ep_ids = features_extraction.load_batch_data(config, data_from_test_file, batch_size=200)
        

#for experiment_number in range(len(all_hyper_parameters)):
                    
            
print('\nTesting: Experiment Number: '+str(experiment_number))
#print('alpha\tbeta_one\tbeta_two\thidden_units\thidden_layers\tdrop_out\tepochs')
#print(custom_parameters)
#print()
        
model = neural_network_testing_v2.load_trained_model(config, custom_parameters, data_from_test_file, experiment_number, time_window, imputation, change, time_stamp)
        

decoding_details = neural_network_testing_v2.testing_neural_network_model(batch_size, config, data_from_test_file, model, experiment_number, all_input_seq_list, all_output_seq_list, all_max_seq_length, batch_wise_ep_ids, time_window, imputation, change, time_stamp)
            
#print(str(time_window)+' hours '+imputation+' '+change+' testing is done')    
   
#'''      
#print ('\nTotal time needed to finish (HH:MM:SS): ', datetime.now()-start_time)
