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
import features_extraction
import neural_network_training, neural_network_architecture
import pickle
import os
from datetime import datetime
import sys
import shutil

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
    
config = configparser.ConfigParser()
config.read('training_configuration.cfg')


#create the directory if it does not exist
save_dir = config.get('directories', 'exp_dir')


if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
    #shutil.rmtree(save_dir)





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


print('\nTraining: Experiment Number: '+str(experiment_number))
print()
print('alpha\tbeta_one\tbeta_two\thidden_units\thidden_layers\tdrop_out\tepochs')
print(custom_parameters)
print()

#training
data_from_train_file = features_extraction.load_train_data(time_window, imputation, config, clinicalBERT, embedding_type)

model = neural_network_training.training_neural_network_model(batch_size, config, data_from_train_file, custom_parameters, experiment_number, time_window, imputation, all_pos_all_neg, change, time_stamp)


#print(str(time_window)+'_hours_'+imputation+'_'+change+'_training_done') 

    
#print ('\nTotal time needed to finish (HH:MM:SS): ', datetime.now()-start_time)
