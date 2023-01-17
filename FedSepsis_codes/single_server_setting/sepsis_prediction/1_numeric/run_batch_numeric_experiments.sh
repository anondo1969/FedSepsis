#!/bin/sh

#author: Mahbub Ul Alam (mahbub@dsv.su.se)
#version: 1.0+
#copyright: Copyright (c) Mahbub Ul Alam (mahbub@dsv.su.se)
#license : MIT License


#sh run_batch_numeric_experiments.sh -e 1 -b 1000 -s episode_wise_precision &>> log_1_experiment_numeric.txt &


#['experiment_number', 'time_window', 'training_label_change',
#'imputation', 'settings', 'range_limit', 'earliness',
#       'episode_wise_accuracy', 'episode_wise_true_positives',
#       'episode_wise_true_negatives', 'episode_wise_false_positives',
#       'episode_wise_false_negatives', 'episode_wise_recall',
#       'episode_wise_precision', 'episode_wise_f1_score',
#       'episode_wise_AU_ROC', 'episode_wise_AU_PRC']

date=`date +"%Y-%m-%d_%T"`

while getopts e:b:s: flag
do
    case "${flag}" in
        e) total_experiments=${OPTARG};;
        b) batch_size=${OPTARG};;
        s) sorting_column=${OPTARG};;
    esac
done
echo "total_experiments: $total_experiments";
echo "batch_size: $batch_size";
echo "sorting_column: $sorting_column";
echo "experiment starting time: $date";


cd 1_training_neural_network_using_training_data/

python3 training_main.py $total_experiments $batch_size $date

echo '\nTraining is done\n'

cd ..

cd 2_applying_the_trained_model_using_tuning_data_in_neural_network/

python3 tuning_main.py $total_experiments $batch_size $date

echo '\nTuning is done\n'

cd ..

cd 3_evaluation_on_predicted_performance_using_tuning_data/

python3 evaluation_tuning_main.py $total_experiments $batch_size $date

echo '\nTuning evaluation is done\n'

cd ..

cd 4_applying_the_trained_model_using_test_data_in_neural_network/

python3 testing_main.py $total_experiments $batch_size $sorting_column $date

echo '\nTesting is done\n'

cd ..

cd 5_evaluation_on_predicted_performance_using_test_data/

python3 evaluation_testing_main.py $total_experiments $batch_size $sorting_column $date

echo '\nTesting evaluation is done\n'


