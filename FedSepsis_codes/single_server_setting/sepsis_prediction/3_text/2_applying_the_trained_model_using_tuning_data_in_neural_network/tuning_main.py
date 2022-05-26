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
from datetime import datetime
import sys

start_time=datetime.now()

total_experiments = int(sys.argv[1])
batch_size = int(sys.argv[2])
clinicalBERT = sys.argv[3]
embedding_type = sys.argv[4]
time_stamp = sys.argv[5]
all_pos_all_neg=int(sys.argv[6])

#if all_pos_all_neg==1:
#    change = 'with_train_label_change'
#else:
#    change = 'without_train_label_change'

#HYPER_PARAMETERS_TUNING = True


#read config file
config = configparser.ConfigParser()
config.read('tuning_configuration.cfg')


#create the directory if it does not exist
save_dir = config.get('directories', 'exp_dir')
data_dir = config.get('directories', 'data_dir')


all_hyper_parameters = neural_network_architecture.hyper_parameter_tuner(config, total_experiments)


#for all_pos_all_neg in [1, 0]:
    
if all_pos_all_neg==1:
    change = 'with_train_label_change'
else:
    change = 'without_train_label_change'

#for time_window in [8,6,4,3,2,1]:
for time_window in [1]:

    #for imputation in ['imputed', 'non_imputed', 'GAN_imputed_10_alpha', 'GAN_imputed_100_alpha']:
        
    #print('\nImputation category: '+imputation+'\n')
    data_from_tune_file = features_extraction.load_data(time_window, config, 'tune', clinicalBERT, embedding_type)

        


    all_input_seq_list, all_output_seq_list, all_max_seq_length, batch_wise_ep_ids = features_extraction.load_batch_data(config, data_from_tune_file, batch_size=200)
        

    for experiment_number in range(len(all_hyper_parameters)):
        
            
        hyper_parameters = all_hyper_parameters[experiment_number]
            
        print('Experiment Number: '+str(experiment_number+1))
        #print('alpha\tbeta_one\tbeta_two\thidden_units\thidden_layers\tdrop_out\tepochs')
        #print(hyper_parameters)
        #print()
        
        model = neural_network_testing_v2.load_trained_model(config, hyper_parameters, data_from_tune_file, experiment_number+1, time_window, change, time_stamp)
        

        decoding_details = neural_network_testing_v2.testing_neural_network_model(batch_size, config, data_from_tune_file, model, experiment_number+1, all_input_seq_list, all_output_seq_list, all_max_seq_length, batch_wise_ep_ids, time_window, change, time_stamp)
            
            
        #print(str(time_window)+' hours '+imputation+' '+change+' tuning is done')    
   
#'''      
print ('\nTotal time needed to finish (HH:MM:SS): ', datetime.now()-start_time)
