#!/bin/sh

#sh run_all_text_experiments.sh -e 1 -b 1

while getopts e:b: flag
do
    case "${flag}" in
        e) total_experiments=${OPTARG};;
        b) batch_size=${OPTARG};;
    esac
done
echo "total_experiments: $total_experiments";
echo "batch_size: $batch_size";

clinicalBERT="Huang Emily"
clinicalBERT_type="short long"
label_change="1 0"
#imputation="GAN_imputed_100_alpha GAN_imputed_10_alpha imputed non_imputed"

for BERT in $clinicalBERT
do
    for BERT_type in $clinicalBERT_type
    do 
    	for change in $label_change
    	do
    		
        	#echo "$BERT $BERT_type $change"
        	sh run_batch_text_experiments.sh -e $total_experiments -b $batch_size -c $BERT -t $BERT_type -s episode_wise_precision -l $change
        	
       done
    done
done


