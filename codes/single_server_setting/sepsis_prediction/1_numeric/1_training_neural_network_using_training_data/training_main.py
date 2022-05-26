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
import neural_network_training, neural_network_architecture
import pickle
import os
from datetime import datetime
import sys
import shutil

start_time=datetime.now()

total_experiments = int(sys.argv[1])
batch_size = int(sys.argv[2])
time_stamp = sys.argv[3]
    
config = configparser.ConfigParser()
config.read('training_configuration.cfg')


#create the directory if it does not exist
save_dir = config.get('directories', 'exp_dir')


if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
    #shutil.rmtree(save_dir)





data_dir = config.get('directories', 'data_dir')

all_hyper_parameters = neural_network_architecture.hyper_parameter_tuner(config, total_experiments)

for all_pos_all_neg in [1, 0]:
    
    if all_pos_all_neg==1:
        change = 'with_train_label_change'
        
    else:
        change = 'without_train_label_change'
        
    print('\n'+change+':\n')
    

    #for time_window in [8,6,4,3,2,1]:
    for time_window in [1]:
        
        for imputation in ['imputed', 'non_imputed', 'GAN_imputed_10_alpha', 'GAN_imputed_100_alpha']:
            
            print('\nImputation category: '+imputation+'\n')
            #try:
            data_from_train_file = features_extraction.load_train_data(time_window, imputation, config)

        
            #except:
                #features_extraction.show_error(save_dir, 'data_from_train_file')

        
       
            for experiment_number in range(len(all_hyper_parameters)):
            
                hyper_parameters = all_hyper_parameters[experiment_number]
            
                print('Experiment Number: '+str(experiment_number+1))
        
                #training
        
                model = neural_network_training.training_neural_network_model(batch_size, config, data_from_train_file, hyper_parameters, experiment_number+1, time_window, imputation, all_pos_all_neg, change, time_stamp)
            
        
                #print(str(time_window)+'_hours_'+imputation+'_'+change+'_training_done') 
            
        
#'''      
print ('\nTotal time needed to finish (HH:MM:SS): ', datetime.now()-start_time)
