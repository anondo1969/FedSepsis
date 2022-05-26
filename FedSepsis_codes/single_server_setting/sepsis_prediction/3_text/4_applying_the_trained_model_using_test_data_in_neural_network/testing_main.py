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
import pandas as pd

start_time=datetime.now()

total_experiments = int(sys.argv[1])
batch_size = int(sys.argv[2])
clinicalBERT = sys.argv[3]
embedding_type = sys.argv[4]
sorting_column = sys.argv[5]
time_stamp = sys.argv[6]
all_pos_all_neg=int(sys.argv[7])

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

evaluation_dir = config.get('directories', 'evaluation_dir')
#information_file_name = config.get('general', 'information_file_name')
information_file_name = config.get('general', 'information_file_name')+'_'+time_stamp+'.tsv'

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
tune_df = pd.read_csv(evaluation_dir+information_file_name, sep='\t')
#imputation_list = tune_df['imputation'].unique().tolist()


imputation_df = tune_df.loc[(tune_df['imputation'] == 'no') & (tune_df['settings'] == 1) & (tune_df['range_limit'] == 0)]
    
sorted_imputation_df = imputation_df.sort_values(by=[sorting_column], ascending=False)
    
best_tune_experiment=sorted_imputation_df.iloc[0][0]


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
    data_from_test_file = features_extraction.load_data(time_window, config, 'test', clinicalBERT, embedding_type)

        


    all_input_seq_list, all_output_seq_list, all_max_seq_length, batch_wise_ep_ids = features_extraction.load_batch_data(config, data_from_test_file, batch_size=200)
        

    #for experiment_number in range(len(all_hyper_parameters)):
        
    experiment_number = best_tune_experiment-1
    hyper_parameters = all_hyper_parameters[experiment_number]
                
            
    print('Experiment Number: '+str(experiment_number+1))
    #print('alpha\tbeta_one\tbeta_two\thidden_units\thidden_layers\tdrop_out\tepochs')
    #print(hyper_parameters)
    #print()
        
    model = neural_network_testing_v2.load_trained_model(config, hyper_parameters, data_from_test_file, experiment_number+1, time_window, change, time_stamp)
        

    decoding_details = neural_network_testing_v2.testing_neural_network_model(batch_size, config, data_from_test_file, model, experiment_number+1, all_input_seq_list, all_output_seq_list, all_max_seq_length, batch_wise_ep_ids, time_window, change, time_stamp)
            
            
    #print(str(time_window)+' hours '+imputation+' '+change+' testing is done')    
   
#'''      
print ('\nTotal time needed to finish (HH:MM:SS): ', datetime.now()-start_time)
