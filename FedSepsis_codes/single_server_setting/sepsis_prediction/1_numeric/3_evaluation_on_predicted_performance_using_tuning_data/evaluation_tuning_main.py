'''
@author: Mahbub Ul Alam (mahbub@dsv.su.se)
@version: 1.0+
@copyright: Copyright (c) Mahbub Ul Alam (mahbub@dsv.su.se)
@license : MIT License
'''

'''@file early_sepsis_detector.py
run this file to go through the complete neural net training, testing and evaluating procedures, 
look at the 'early_sepsis_detector_configuration.cfg' file to modify the settings'''

from six.moves import configparser
import evaluate_results
import os
from datetime import datetime
import sys

start_time=datetime.now()

total_experiments = int(sys.argv[1])
batch_size = int(sys.argv[2])
time_stamp = sys.argv[3]

#read config file
config = configparser.ConfigParser()
config.read('evaluation_tuning_configuration.cfg')

#if os.path.exists(config.get('directories', 'evaluation_dir')+config.get('general', 'information_file_name')):
    #os.remove(config.get('directories', 'evaluation_dir')+config.get('general', 'information_file_name'))
    
all_hyper_parameters = evaluate_results.hyper_parameter_tuner(config, total_experiments)

header = True

for change in ['with_train_label_change', 'without_train_label_change']:

#for time_window in [8,6,4,3,2,1]:
    for time_window in [1]:
        for imputation in ['imputed', 'non_imputed', 'GAN_imputed_10_alpha', 'GAN_imputed_100_alpha']:
           

            for value in range(1, total_experiments+1):
    
                decoding_details = evaluate_results.load_decoding_details(config, value, time_window, imputation, change, time_stamp)
        
                #print('Experiment: '+str(value))
        
                #evaluation
            
                timestep_wise_y_true_and_y_predicts =    evaluate_results.remove_dummy_pad_value_zero_from_timestep_wise_y_true_and_y_predict(decoding_details, config)
                
                
        
                for  settings in range(1,4):
                    for range_limit in [0,24,48]:
        
                        episodes_based_y_true_and_y_predicts = evaluate_results.compute_episodes_wise_y_true_and_y_predicts (timestep_wise_y_true_and_y_predicts, decoding_details, config, settings, range_limit, time_window)
                        
             
           

                        episode_wise_evaluation_scores = evaluate_results.compute_episode_wise_evaluation_scores (episodes_based_y_true_and_y_predicts, config)
                
                
            
                        earliness = evaluate_results.compute_episodes_wise_earliness(episodes_based_y_true_and_y_predicts, config, settings, range_limit, time_window)
                
                        evaluate_results.write_experiment_information(config, earliness, episode_wise_evaluation_scores, settings, range_limit, value, time_window, imputation, change, header, batch_size, time_stamp, all_hyper_parameters)
                        
                        header = False
                    
                        '''
                        print('\nEpisode wise results: ')
                        print('settings: '+str(settings)+' range_limit: '+str(range_limit))
                        print('accuracy\ttp\ttn\tfp\tfn\trecall\tprecision\tf1_score\tAU_ROC\tAU_PRC')
                        print(episode_wise_evaluation_scores)
                        print('earliness: '+str(earliness))
                        '''
            print(str(time_window)+' hours '+imputation+' '+change+' tuning scoring is done')
                    

                
        
                 

print ('\nTotal time needed to finish (HH:MM:SS): ', datetime.now()-start_time)
