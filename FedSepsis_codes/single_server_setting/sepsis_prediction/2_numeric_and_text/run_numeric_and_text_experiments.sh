#!/bin/sh


#sh run_numeric_and_text_experiments.sh -x 27 -b 500 -h 800 -H 3 -d 0 -e 3 -i GAN_imputed_100_alpha -c Huang -C long

# Huang -t short

# -t long

#['experiment_number', 'time_window', 'training_label_change',
#'imputation', 'settings', 'range_limit', 'earliness',
#       'episode_wise_accuracy', 'episode_wise_true_positives',
#       'episode_wise_true_negatives', 'episode_wise_false_positives',
#       'episode_wise_false_negatives', 'episode_wise_recall',
#       'episode_wise_precision', 'episode_wise_f1_score',
#       'episode_wise_AU_ROC', 'episode_wise_AU_PRC']


date=`date +"%Y-%m-%d_%T"`

#date="2022-02-15_13:07:54"

while getopts x:b:h:H:d:e:i:c:C:s: flag
do
    case "${flag}" in
         x) experiment_number=${OPTARG};;
         b) batch_size=${OPTARG};;
         h) hidden_units=${OPTARG};;
         H) hidden_layers=${OPTARG};;
         d) drop_out=${OPTARG};;
         e) epochs=${OPTARG};;
         i) imputation=${OPTARG};;
         c) clinicalBERT=${OPTARG};;
         C) embedding_type=${OPTARG};;
    esac
done


echo "";
echo "experiment_number: $experiment_number";
echo "hidden_units: $hidden_units";
echo "hidden_layers: $hidden_layers";
echo "drop_out: $drop_out";
echo "epochs: $epochs";
echo "imputation: $imputation";
echo "batch_size: $batch_size";
echo "clinicalBERT: $clinicalBERT";
echo "embedding_type: $embedding_type";
echo "experiment starting time: $date";

cd 1_training_neural_network_using_training_data/


python3 training_main.py $hidden_units $hidden_layers $drop_out $epochs $imputation $experiment_number $batch_size $clinicalBERT $embedding_type $date

echo '\nTraining is done\n'

cd ..


cd 2_applying_the_trained_model_using_tuning_data_in_neural_network/

python3 tuning_main.py  $hidden_units $hidden_layers $drop_out $epochs $imputation $experiment_number $batch_size $clinicalBERT $embedding_type $date

echo '\nTuning is done\n'

cd ..

cd 3_evaluation_on_predicted_performance_using_tuning_data/

python3 evaluation_tuning_main.py $experiment_number $imputation $batch_size $clinicalBERT $embedding_type $date

echo '\nTuning evaluation is done\n'

cd ..

cd 4_applying_the_trained_model_using_test_data_in_neural_network/

python3 testing_main.py  $hidden_units $hidden_layers $drop_out $epochs $imputation $experiment_number $batch_size $clinicalBERT $embedding_type $date

echo '\nTesting is done\n'

cd ..

cd 5_evaluation_on_predicted_performance_using_test_data/

python3 evaluation_testing_main.py  $total_experiments  $experiment_number $imputation $batch_size $clinicalBERT $embedding_type $date

echo '\nTesting evaluation is done\n'





