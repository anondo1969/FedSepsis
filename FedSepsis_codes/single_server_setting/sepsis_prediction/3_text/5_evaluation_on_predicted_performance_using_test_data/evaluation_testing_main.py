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
import evaluate_results
from datetime import datetime
import sys
import pandas as pd

start_time=datetime.now()

total_experiments = int(sys.argv[1])
batch_size = int(sys.argv[2])
clinicalBERT = sys.argv[3]
embedding_type = sys.argv[4]
sorting_column = sys.argv[5]
time_stamp = sys.argv[6]
all_pos_all_neg=int(sys.argv[7])

#read config file
config = configparser.ConfigParser()
config.read('evaluation_testing_configuration.cfg')

evaluation_dir = config.get('directories', 'evaluation_dir')
#tune_information_file_name = config.get('general', 'tune_information_file_name')
tune_information_file_name = config.get('general', 'tune_information_file_name')+'_'+time_stamp+'.tsv'
all_hyper_parameters = evaluate_results.hyper_parameter_tuner(config, total_experiments)
#sorting_column = 'episode_wise_AU_ROC'
'''
['experiment_number', 'time_window', 'training_label_change',
'imputation', 'settings', 'range_limit', 'earliness',
       'episode_wise_accuracy', 'episode_wise_true_positives',
       'episode_wise_true_negatives', 'episode_wise_false_positives',
       'episode_wise_false_negatives', 'episode_wise_recall',
       'episode_wise_precision', 'episode_wise_f1_score',
       'episode_wise_AU_ROC', 'episode_wise_AU_PRC']
'''
tune_df = pd.read_csv(evaluation_dir+tune_information_file_name, sep='\t')
imputation_list = tune_df['imputation'].unique().tolist()


                       
imputation_df = tune_df.loc[(tune_df['imputation'] == 'no') & (tune_df['settings'] == 1) & (tune_df['range_limit'] == 0)]
    
sorted_imputation_df = imputation_df.sort_values(by=[sorting_column], ascending=False)
    
best_tune_experiment=sorted_imputation_df.iloc[0][0]


#if os.path.exists(config.get('directories', 'evaluation_dir')+config.get('general', 'information_file_name')):
    #os.remove(config.get('directories', 'evaluation_dir')+config.get('general', 'information_file_name'))

header = True

if all_pos_all_neg==1:
    change = 'with_train_label_change'
else:
    change = 'without_train_label_change'

#for change in ['with_train_label_change', 'without_train_label_change']:

#for time_window in [8,6,4,3,2,1]:
for time_window in [1]:
    #for imputation in ['imputed', 'non_imputed', 'GAN_imputed_10_alpha', 'GAN_imputed_100_alpha']:
           

    #for value in range(1, total_experiments+1):
    value = best_tune_experiment
    
    decoding_details = evaluate_results.load_decoding_details(config, value, time_window, change, time_stamp)
        
    #print('Experiment: '+str(value))
        
    #evaluation
            
    timestep_wise_y_true_and_y_predicts =    evaluate_results.remove_dummy_pad_value_zero_from_timestep_wise_y_true_and_y_predict(decoding_details, config)
                
                
        
    for  settings in range(1,4):
        for range_limit in [0,24,48]:
        
            episodes_based_y_true_and_y_predicts = evaluate_results.compute_episodes_wise_y_true_and_y_predicts (timestep_wise_y_true_and_y_predicts, decoding_details, config, settings, range_limit, time_window)
            episode_wise_evaluation_scores = evaluate_results.compute_episode_wise_evaluation_scores (episodes_based_y_true_and_y_predicts, config)
                
                
            
            earliness = evaluate_results.compute_episodes_wise_earliness(episodes_based_y_true_and_y_predicts, config, settings, range_limit, time_window)
                
            evaluate_results.write_experiment_information(config, earliness, episode_wise_evaluation_scores, settings, range_limit, value, time_window, change, header, clinicalBERT, embedding_type, batch_size, time_stamp, all_hyper_parameters)
                    
            header = False
            '''
            print('\nEpisode wise results: ')
            print('settings: '+str(settings)+' range_limit: '+str(range_limit))
            print('accuracy\ttp\ttn\tfp\tfn\trecall\tprecision\tf1_score\tAU_ROC\tAU_PRC')
            print(episode_wise_evaluation_scores)
            print('earliness: '+str(earliness))
            '''

print(str(time_window)+' hours testing scoring is done')
                    

                
        
                 

print ('\nTotal time needed to finish (HH:MM:SS): ', datetime.now()-start_time)
