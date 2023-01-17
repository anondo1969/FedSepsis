'''
@author: Mahbub Ul Alam (mahbub@dsv.su.se)
@version: 1.0+
@copyright: Copyright (c) Mahbub Ul Alam (mahbub@dsv.su.se)
@license : MIT License
'''
import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch

from torch.utils.data import WeightedRandomSampler, DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))


def local_dataloader(args, data_type, client_id):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv(
        args.data_dir + '/' + args.imputation + '_' + data_type + '_' + str(args.time_window) + "_hours_data.csv")

    label_list = []

    episode_id_list = df['icustay_id'].unique().tolist()

    pos_df = pd.read_csv(args.data_dir + '/' + 'case_total_data_' + str(args.time_window) + '_hours.csv')

    pos_episode_id_list = pos_df['icustay_id'].unique().tolist()

    for episode_id in episode_id_list:
        if episode_id in pos_episode_id_list:
            label_list.append(1)
        else:
            label_list.append(0)

    all_episode_id_list = np.array(episode_id_list)
    all_sepsis_patient_label_list = np.array(label_list)

    if data_type == 'train':

        pos_idx = np.where(all_sepsis_patient_label_list == 1)[0]
        neg_idx = np.where(all_sepsis_patient_label_list == 0)[0]

        pos_episode_id_list = all_episode_id_list[pos_idx]
        neg_episode_id_list = all_episode_id_list[neg_idx]

        splitted_pos_episode_id_list = np.array_split(pos_episode_id_list, args.client_num_in_total)
        splitted_neg_episode_id_list = np.array_split(neg_episode_id_list, args.client_num_in_total)

        pos_sepsis_patient_label_list = all_sepsis_patient_label_list[pos_idx]
        neg_sepsis_patient_label_list = all_sepsis_patient_label_list[neg_idx]

        splitted_pos_sepsis_patient_label_list = np.array_split(pos_sepsis_patient_label_list, args.client_num_in_total)
        splitted_neg_sepsis_patient_label_list = np.array_split(neg_sepsis_patient_label_list, args.client_num_in_total)

        client_episode_id_list = np.append(splitted_pos_episode_id_list[client_id],
                                           splitted_neg_episode_id_list[client_id])
        client_sepsis_patient_label_list = np.append(splitted_pos_sepsis_patient_label_list[client_id],
                                                     splitted_neg_sepsis_patient_label_list[client_id])

        # logging.info("Total batches = %d" % (len(client_episode_id_list) // args.batch_size))
        print("Total batches = %d" % (int(len(client_episode_id_list) / args.batch_size) + 1))

        # print("Total batches = %d" % (int(float(len(client_episode_id_list))  / float(args.batch_size))))

        loader = create_train_data_upsampled_numbers(client_episode_id_list, client_sepsis_patient_label_list,
                                                     args.batch_size)

        data_num = len(client_episode_id_list)

    else: # test data

        data_num = len(all_episode_id_list)
        target = torch.from_numpy(all_sepsis_patient_label_list).long()
        data = torch.from_numpy(all_episode_id_list).long()

        dataset = TensorDataset(data, target)
        # print("length of dataset "+str(len(dataset)))
        num_workers=0
        if device == 'cpu':
            num_workers=1
        loader = DataLoader(dataset, batch_size=args.test_batch_size, num_workers=num_workers)


    return loader, data_num


def create_train_data_upsampled_numbers(data, target, batch_size):
    '''
    Purpose:
        Upsampling the training episodes in the batch to make the data more balanced (50-50)

    Args:
        data : episode ids
        target: labels for each episode id
        class_sample_count : frequency of episodes for each class
        batch_size : total number of episodes for each batch

    Returns:
        train_loader: a DataLoader object from torch.utils.data
    '''

    # print ('target train 0/1: {}/{}'.format(len(np.where(target == 0)[0]), len(np.where(target == 1)[0])))
    # print(target)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class_sample_count = np.array(
        [len(np.where(target == t)[0]) for t in np.unique(target)])

    weight = 1. / class_sample_count
    # weight[0] = 0

    samples_weight = np.array([weight[t] for t in target])
    # samples_weight = [weight[t] for t in target]

    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = weight[target]
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    target = torch.from_numpy(target).long()
    data = torch.from_numpy(data).long()

    train_dataset = TensorDataset(data, target)

    num_workers=0
    if device == 'cpu':
        num_workers=1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers)

    return train_loader


def load_all_data(args, data_type):
    df = pd.read_csv(
        args.data_dir + '/' + args.imputation + '_' + data_type + '_' + str(args.time_window) + "_hours_data.csv")

    row_df = pd.read_csv(args.data_dir + '/' + data_type + '_data_' + str(args.time_window) + "_hours.csv")

    # df = pd.read_csv(args.data_dir + '/' + args.imputation + '_tune_' + str(args.time_window) + "_hours_data.csv")

    # row_df = pd.read_csv(args.data_dir + '/' + 'tune_data_' + str(args.time_window) + "_hours.csv")

    # print(row_df.columns)

    row_df = row_df[['row_id', 'icustay_id']]  # [row_df.columns & ['ROW_ID','icustay_id']]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.clinicalBERT == 'Emily':
        embedding_dict = torch.load(
            args.data_dir + '/' + 'clinicalBERT_Emily_torch_' + args.embedding_type + '_embedding_dict',
            map_location=device)
    else:
        embedding_dict = torch.load(
            args.data_dir + '/' + 'clinicalBERT_Huang_torch_' + args.embedding_type + '_embedding_dict',
            map_location=device)

    if args.embedding_type == 'short':
        embedding_size = 768
    else:
        embedding_size = 768 * 4

    if args.experiment_category == 'non-text':
        input_size = df.shape[1] - 2 # - episode id, - sepsisBin
    elif args.experiment_category == 'multi-modal':
        input_size = df.shape[1] - 2 + embedding_size # - episode id, - sepsisBin
    elif args.experiment_category == 'text':
        input_size = embedding_size

    outputs = {
        'class_label_values': [1, 2],
        'num_classes': 2,
        'total_data': df,
        'input_size': input_size,
        'row_df': row_df,
        'embedding_dict': embedding_dict,
        'embedding_size': embedding_size
    }

    return outputs


def load_batch_data_non_text(ep_id_list, whole_data, label_column, all_pos_all_neg):
    '''
    Purpose:
        Loading the batch data for training neural networks

    Args:
        ep_id_list : the list of unique episode ids
        whole_data : all the needed columns in panadas data frame format
        label_column : the name of the label column

    Returns:
        input_seq_list : input data in sequence for the batch
        output_seq_list : output data in sequence for the batch
        output_string_seq_dict : output labels in sequence for the batch where the key is the episode id
        max_seq_length : maximum length of the episodes time steps, it is variable for each episode
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = whole_data

    # ep_ids = ep_id_list.to(device).numpy() #converting from torch
    ep_ids = ep_id_list.cpu().numpy()  # converting from torch

    # print("Total episodes: " + str(len(ep_ids)))

    input_seq_list = []
    output_seq_list = []

    # exp_dir = '/Volumes/mahbub_veracrypt_mac/sample_test/'
    # label_file = open(exp_dir+'label_file_train.txt','a')

    for ep_id in ep_ids:

        seq = df.loc[df['icustay_id'] == ep_id]

        output_seq = seq[label_column].values.tolist()

        # label_file.write('\n'+str(output_seq))

        # code to make all positive or negative
        if all_pos_all_neg == 1:
            if output_seq[-1] == 2:  # positive case
                new_output_seq = [2 for value in output_seq]
            elif output_seq[-1] == 1:  # negative case
                new_output_seq = [1 for value in output_seq]

            output_seq = new_output_seq

        input_seq = seq.drop(columns=[label_column, 'icustay_id']).values.tolist()


        input_seq_torch = torch.tensor(input_seq).to(device)
        output_seq_torch = torch.tensor(output_seq).to(device)

        input_seq_list.append(input_seq_torch)
        output_seq_list.append(output_seq_torch)

    max_seq_length = max(list(map(len, output_seq_list)))

    if max_seq_length == 0:
        print('training errorrrrrrr')

    return input_seq_list, output_seq_list, max_seq_length


def load_batch_data_text(ep_id_list, whole_data, row_df, embedding_dict, embedding_size, label_column,
                          all_pos_all_neg):
    '''
    Purpose:
        Loading the batch data for training neural networks

    Args:
        ep_id_list : the list of unique episode ids
        whole_data : all the needed columns in panadas data frame format
        label_column : the name of the label column

    Returns:
        input_seq_list : input data in sequence for the batch
        output_seq_list : output data in sequence for the batch
        output_string_seq_dict : output labels in sequence for the batch where the key is the episode id
        max_seq_length : maximum length of the episodes time steps, it is variable for each episode
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = whole_data

    ep_ids = ep_id_list.cpu().numpy()  # converting from torch



    input_seq_list = []
    output_seq_list = []

    for ep_id in ep_ids:

        # concatenate it here
        seq = df.loc[df['icustay_id'] == ep_id]

        row_ids = row_df.loc[row_df['icustay_id'] == ep_id]['row_id'].tolist()

        # print(row_ids)
        # print(ep_id)
        embeddings_per_row = []

        for row_id in row_ids:

            try:
                row_embedding = embedding_dict[row_id]  # .cpu()

            except KeyError:
                row_embedding = torch.zeros(embedding_size)  # .cpu()

            embeddings_per_row.append(row_embedding)

        output_seq = seq[label_column].values.tolist()


        # code to make all positive or negative
        if all_pos_all_neg == 1:
            if output_seq[-1] == 2:  # positive case
                new_output_seq = [2 for value in output_seq]
            elif output_seq[-1] == 1:  # negative case
                new_output_seq = [1 for value in output_seq]

            output_seq = new_output_seq


        seq = seq.drop(columns=[label_column, 'icustay_id']).values.tolist()

        input_seq = []

        embedding_index = -1

        for one_sequence in seq:

            embedding_index += 1

            embedding_sequence_tensor = embeddings_per_row[embedding_index]

            input_seq.append(embedding_sequence_tensor)

        input_seq_torch = torch.stack(input_seq).to(device)
        output_seq_torch = torch.tensor(output_seq).to(device)

        input_seq_list.append(input_seq_torch)
        output_seq_list.append(output_seq_torch)

    max_seq_length = max(list(map(len, output_seq_list)))

    if max_seq_length == 0:
        print('training errorrrrrrr')

    return input_seq_list, output_seq_list, max_seq_length


def load_batch_data_multi_modal(ep_id_list, whole_data, row_df, embedding_dict, embedding_size, label_column,
                    all_pos_all_neg):
    '''
    Purpose:
        Loading the batch data for training neural networks

    Args:
        ep_id_list : the list of unique episode ids
        whole_data : all the needed columns in panadas data frame format
        label_column : the name of the label column

    Returns:
        input_seq_list : input data in sequence for the batch
        output_seq_list : output data in sequence for the batch
        output_string_seq_dict : output labels in sequence for the batch where the key is the episode id
        max_seq_length : maximum length of the episodes time steps, it is variable for each episode
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = whole_data

    ep_ids = ep_id_list.cpu().numpy()  # converting from torch

    input_seq_list = []
    output_seq_list = []

    for ep_id in ep_ids:

        # concatenate it here
        seq = df.loc[df['icustay_id'] == ep_id]

        row_ids = row_df.loc[row_df['icustay_id'] == ep_id]['row_id'].tolist()

        embeddings_per_row = []

        for row_id in row_ids:

            try:
                row_embedding = embedding_dict[row_id]  # .cpu()

            except KeyError:
                row_embedding = torch.zeros(embedding_size)  # .cpu()

            embeddings_per_row.append(row_embedding)

        output_seq = seq[label_column].values.tolist()

        # code to make all positive or negative
        if all_pos_all_neg == 1:
            if output_seq[-1] == 2:  # positive case
                new_output_seq = [2 for value in output_seq]
            elif output_seq[-1] == 1:  # negative case
                new_output_seq = [1 for value in output_seq]

            output_seq = new_output_seq

        seq = seq.drop(columns=[label_column, 'icustay_id']).values.tolist()

        input_seq = []

        embedding_index = -1

        for one_sequence in seq:
            one_sequence_tensor = torch.tensor(one_sequence).to(device)

            embedding_index += 1

            embedding_sequence_tensor = embeddings_per_row[embedding_index]

            one_numeric_text_sequence = torch.cat([one_sequence_tensor, embedding_sequence_tensor], dim=0).to(device)

            input_seq.append(one_numeric_text_sequence)

        input_seq_torch = torch.stack(input_seq).to(device)
        output_seq_torch = torch.tensor(output_seq).to(device)
        input_seq_list.append(input_seq_torch)
        output_seq_list.append(output_seq_torch)

    max_seq_length = max(list(map(len, output_seq_list)))

    if max_seq_length == 0:
        print('training errorrrrrrr')

    return input_seq_list, output_seq_list, max_seq_length


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        # logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./../data/SEPSIS',
                        help='data directory')
    parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 64)')
    args = parser.parse_args()
    logging.info(args)

    dataset = local_dataloader(args)
    [train_data_num, test_data_num, train_data_global, test_data_global,
     train_data_local_num_dict, train_data_local_dict, test_data_local_dict] = dataset
    print(dataset)
