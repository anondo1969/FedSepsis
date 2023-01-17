#!/bin/sh

#author: Mahbub Ul Alam (mahbub@dsv.su.se)
#version: 1.0+
#copyright: Copyright (c) Mahbub Ul Alam (mahbub@dsv.su.se)
#license : MIT License


#sh run_batch_text_experiments.sh -e 50 -b 500 -c Huang -t short -s episode_wise_precision -l 1 &>> new_huang_50_500_short.txt &

#['imputed', 'non_imputed', 'GAN_imputed_10_alpha', 'GAN_imputed_100_alpha']
# Huang -t short imputation

# -t long

#['experiment_number', 'time_window', 'training_label_change',
#'imputation', 'settings', 'range_limit', 'earliness',
#       'episode_wise_accuracy', 'episode_wise_true_positives',
#       'episode_wise_true_negatives', 'episode_wise_false_positives',
#       'episode_wise_false_negatives', 'episode_wise_recall',
#       'episode_wise_precision', 'episode_wise_f1_score',
#       'episode_wise_AU_ROC', 'episode_wise_AU_PRC']

date=`date +"%Y-%m-%d_%T"`

while getopts e:b:c:t:s:l:i: flag
do
    case "${flag}" in
        e) total_experiments=${OPTARG};;
        b) batch_size=${OPTARG};;
        c) clinicalBERT=${OPTARG};;
        t) embedding_type=${OPTARG};;
        s) sorting_column=${OPTARG};;
        l) label_change=${OPTARG};;
    esac
done
echo "total_experiments: $total_experiments";
echo "batch_size: $batch_size";
echo "clinicalBERT: $clinicalBERT";
echo "embedding_type: $embedding_type";
echo "sorting_column: $sorting_column";
echo "label_change: $label_change";
echo "experiment starting time: $date";

cd 1_training_neural_network_using_training_data/

python3 training_main.py $total_experiments $batch_size $clinicalBERT $embedding_type $date $label_change

echo '\nTraining is done\n'

cd ..

cd 2_applying_the_trained_model_using_tuning_data_in_neural_network/

python3 tuning_main.py $total_experiments $batch_size $clinicalBERT $embedding_type $date $label_change

echo '\nTuning is done\n'

cd ..

cd 3_evaluation_on_predicted_performance_using_tuning_data/

python3 evaluation_tuning_main.py $total_experiments $batch_size $clinicalBERT $embedding_type $date $label_change

echo '\nTuning evaluation is done\n'

cd ..

cd 4_applying_the_trained_model_using_test_data_in_neural_network/

python3 testing_main.py $total_experiments $batch_size $clinicalBERT $embedding_type $sorting_column $date $label_change

echo '\nTesting is done\n'

cd ..

cd 5_evaluation_on_predicted_performance_using_test_data/

python3 evaluation_testing_main.py  $total_experiments $batch_size $clinicalBERT $embedding_type $sorting_column $date $label_change

echo '\nTesting evaluation is done\n'





