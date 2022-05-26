'''
@author: Mahbub Ul Alam (mahbub@dsv.su.se)
@date: 25.01.2019
@version: 1.0+
@copyright: Copyright (c)  2019-2020, Mahbub Ul Alam (mahbub@dsv.su.se)
@license : MIT License
'''

import numpy as np
import pickle
from torch.utils.data import WeightedRandomSampler, DataLoader, TensorDataset
import pandas as pd
import torch

        

def load_train_data(time_window, imputation, config):
    
    #train feature loading
    #try:
        
        save_dir = config.get('directories', 'exp_dir')
        
        data_dir = config.get('directories', 'data_dir')
        
        #embedding_data_dir = config.get('directories', 'embedding_data_dir')
        
        #clinicalBERT = config.get('general', 'clinicalBERT')
        
        df = pd.read_csv(data_dir+imputation+'_train_'+str(time_window)+"_hours_data.csv")
        
        
        #row_df = pd.read_csv(data_dir+'train_data_'+str(time_window)+"_hours.csv")
        
        #print(row_df.columns)
        
        #row_df = row_df[['row_id','icustay_id']]#[row_df.columns & ['ROW_ID','icustay_id']]
        
        #embedding_type = config.get('general', 'embedding_type')
        
        #if clinicalBERT == 'Emily':
            #embedding_dict = torch.load(embedding_data_dir+'clinicalBERT_Emily_torch_'+embedding_type+'_embedding_dict')
        #else:
            #embedding_dict = torch.load(embedding_data_dir+'clinicalBERT_Huang_torch_short_embedding_dict')
            
        #if embedding_type == 'short':
            #embedding_size = 768
        #else:
            #embedding_size = 768*4
       
        episode_id_list, sepsis_patient_label_list = create_episode_id_label_list(df, data_dir, time_window)
  
        outputs = {
                'episode_id_list' : episode_id_list,
                'sepsis_patient_label_list' : sepsis_patient_label_list, 
                'class_label_values' : [1, 2], 
                'num_classes' : 2,
                'total_data' : df,
                'input_size' : df.shape[1]-2 # - episode id, - sepsisBin
                #'row_df' : row_df,
                #'embedding_dict' : embedding_dict,
                #'embedding_size' : embedding_size
                }
        
        
        return outputs
     
    #except TypeError:
        #show_error(save_dir, 'train_data file')
        

def create_episode_id_label_list(df, data_dir, time_window):
        
    label_list = []
       
    episode_id_list = df['icustay_id'].unique().tolist()
       
    pos_df = pd.read_csv(data_dir+'case_total_data_'+str(time_window)+'_hours.csv')
       
    pos_episode_id_list = pos_df['icustay_id'].unique().tolist()
        
        
    for episode_id in episode_id_list:
        if episode_id in pos_episode_id_list:
            label_list.append(1)
        else:
            label_list.append(0)
            
                
    return np.array(episode_id_list), np.array(label_list)
        
                

def load_train_batch_data(ep_id_list, whole_data, label_column, all_pos_all_neg):
    
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
    
    df=whole_data
        

    #ep_ids = ep_id_list.to(device).numpy() #converting from torch
    ep_ids = ep_id_list.cpu().numpy() #converting from torch
    
    
    #print("Total episodes: " + str(len(ep_ids)))
    
    
    input_seq_list = []
    output_seq_list = []
    
    
    #exp_dir = '/Volumes/mahbub_veracrypt_mac/sample_test/'
    #label_file = open(exp_dir+'label_file_train.txt','a')
    
    for ep_id in ep_ids:
        
        # concatenate it here
        seq = df.loc[df['icustay_id'] == ep_id]
        
        '''
        row_ids = row_df.loc[row_df['icustay_id'] == ep_id]['row_id'].tolist()
        
        #print(row_ids)
        #print(ep_id)
        embeddings_per_row = []
        
        for row_id in row_ids:
            
            try:
                row_embedding = embedding_dict[row_id].cpu()
                
            except KeyError:
                row_embedding = torch.zeros(embedding_size).cpu()
            
            embeddings_per_row.append(row_embedding)
            
        #seq['text'] = embeddings_per_row
        
        
        '''
        #if len(seq)>1:
              
        output_seq = seq[label_column].values.tolist()
        
        #label_file.write('\n'+str(output_seq))
        
        #code to make all positive or negative
        if all_pos_all_neg==1:
            if output_seq[-1]==2: # positive case
                new_output_seq = [2 for value in output_seq]
            elif output_seq[-1]==1: # negative case
                new_output_seq = [1 for value in output_seq]
            
            output_seq = new_output_seq
            
            #label_file.write('\n'+str(output_seq))
        
        
        
        input_seq = seq.drop(columns=[label_column, 'icustay_id']).values.tolist()
        #print(input_seq)
        #print('done')
        
        #for embedding_value in embeddings_per_row:
        
        '''   
        input_seq=[]
            
        embedding_index = -1
        
        for one_sequence in seq:
            
            one_sequence_tensor = torch.tensor(one_sequence)
            #print(one_sequence_tensor.size())
            
            embedding_index+=1
            
            embedding_sequence_tensor = embeddings_per_row[embedding_index]
            #print(embedding_sequence_tensor.size())
            
            #print(one_sequence_tensor.size())
            
            #print(embedding_sequence_tensor.size())
            
            #torch.cat([a,b], dim=0)
            
            one_numeric_text_sequence = torch.cat([one_sequence_tensor,embedding_sequence_tensor], dim=0)
            
            #print(one_numeric_text_sequence.size())
            
            input_seq.append(one_numeric_text_sequence.cpu().detach().numpy())
        '''
        
        
        input_seq_torch=torch.tensor(input_seq).to(device)
        output_seq_torch=torch.tensor(output_seq).to(device)
        #input_seq_list.append(input_seq)
        #output_seq_list.append(output_seq)
        input_seq_list.append(input_seq_torch)
        output_seq_list.append(output_seq_torch)
        
        
    #label_file.close()   
    
       
    
    max_seq_length=max(list(map(len, output_seq_list)))
    
    if max_seq_length==0:
        print('training errorrrrrrr')
    
    #print('max_seq_length: '+str(max_seq_length))
    
    #min_seq_length = min(list(map(len, output_seq_list)))
            
    #print('min_seq_length: '+str(min_seq_length))
    
    return input_seq_list, output_seq_list, max_seq_length
    #return np.array(input_seq_list, dtype=object), np.array(output_seq_list, dtype=object), max_seq_length

def create_train_data_upsampled_numbers(data, target, batch_size):
    
    '''
    Purpose:
        Upsampling the testing episodes in the batch to make the data more balanced (50-50)

    Args:
        data : episode ids
        target: labels for each episode id
        class_sample_count : frequency of episodes for each class
        batch_size : total number of episodes for each batch
            
    Returns:
        train_loader: a DataLoader object from torch.utils.data
    ''' 
    
    #print ('target train 0/1: {}/{}'.format(len(np.where(target == 0)[0]), len(np.where(target == 1)[0])))
    #print(target)
    
    class_sample_count = np.array(
        [len(np.where(target == t)[0]) for t in np.unique(target)])
        
        
    weight = 1. / class_sample_count
    #weight[0] = 0
    
    
    samples_weight = np.array([weight[t] for t in target])
    #samples_weight = [weight[t] for t in target]
    

    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = weight[target]
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    target = torch.from_numpy(target).long()
    data = torch.from_numpy(data).long()
    
    train_dataset = TensorDataset(data, target)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                             num_workers=1)
                             
    
    return train_loader

def show_error(save_dir, file_name):
    
    print('\nFile or Directory Not Found Error:\n\nDid not find the '+file_name+' in the location '+save_dir+'\n\nMake sure of these followings, \n1. you have the right permission to create the directory in that location\n2. you have enabled your encrypted directory to access it.\n3. The experiment directory '+save_dir+' is the correct one\n4. you have executed the previous module(s) successfully before at least once.\n')

