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

value = int(sys.argv[1])
imputation=sys.argv[2]
batch_size = int(sys.argv[3])
clinicalBERT = sys.argv[4]
embedding_type = sys.argv[5]
time_stamp = sys.argv[6]

#read config file
config = configparser.ConfigParser()
config.read('evaluation_tuning_configuration.cfg')

#if os.path.exists(config.get('directories', 'evaluation_dir')+config.get('directories', 'information_file_name')):
    #os.remove(config.get('directories', 'evaluation_dir')+config.get('directories', 'information_file_name'))

header = True

all_pos_all_neg=1
change = 'with_train_label_change'
time_window=1
    
decoding_details = evaluate_results.load_decoding_details(config, value, time_window, imputation, change, time_stamp)
        
                
            
timestep_wise_y_true_and_y_predicts =    evaluate_results.remove_dummy_pad_value_zero_from_timestep_wise_y_true_and_y_predict(decoding_details, config)
                
                
print('\nEpisode wise results: ')         
for  settings in range(1,4):
    for range_limit in [0,24,48]:
        
        episodes_based_y_true_and_y_predicts = evaluate_results.compute_episodes_wise_y_true_and_y_predicts (timestep_wise_y_true_and_y_predicts, decoding_details, config, settings, range_limit, time_window)
                        
             
           

        episode_wise_evaluation_scores = evaluate_results.compute_episode_wise_evaluation_scores (episodes_based_y_true_and_y_predicts, config)
                
                
            
        earliness = evaluate_results.compute_episodes_wise_earliness(episodes_based_y_true_and_y_predicts, config, settings, range_limit, time_window)
                
        evaluate_results.write_experiment_information(config, earliness, episode_wise_evaluation_scores, settings, range_limit, value, time_window, imputation, change, header, clinicalBERT, embedding_type, batch_size, time_stamp)
                    
        header = False
        print('settings: '+str(settings)+' range_limit: '+str(range_limit))
        print('accuracy\ttp\ttn\tfp\tfn\trecall\tprecision\tf1_score\tAU_ROC\tAU_PRC')
        print()
        print(episode_wise_evaluation_scores)
        print()
        print('earliness: '+str(earliness))
        print()

#print(str(time_window)+' hours '+imputation+' '+change+' tuning scoring is done')
                    

#print ('\nTotal time needed to finish (HH:MM:SS): ', datetime.now()-start_time)
