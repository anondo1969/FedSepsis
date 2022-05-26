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


def load_data(time_window, imputation, config, data_type):
    
    save_dir = config.get('directories', 'exp_dir')
    
    data_dir = config.get('directories', 'data_dir')
    
    #embedding_data_dir = config.get('directories', 'embedding_data_dir')
    
    #clinicalBERT = config.get('general', 'clinicalBERT')
    
    df = pd.read_csv(data_dir+imputation+'_'+data_type+'_'+str(time_window)+"_hours_data.csv")
    
    #tune_data_1_hours.csv
    #row_df = pd.read_csv(data_dir+data_type+'_data_'+str(time_window)+"_hours.csv")
    
    #print(row_df.columns)
    '''
    row_df = row_df[['row_id','icustay_id']]#[row_df.columns & ['ROW_ID','icustay_id']]
    
    embedding_type = config.get('general', 'embedding_type')
    
    if clinicalBERT == 'Emily':
        embedding_dict = torch.load(embedding_data_dir+'clinicalBERT_Emily_torch_'+embedding_type+'_embedding_dict')
    else:
        embedding_dict = torch.load(embedding_data_dir+'clinicalBERT_Huang_torch_short_embedding_dict')
        
    if embedding_type == 'short':
        embedding_size = 768
    else:
        embedding_size = 768*4
    '''            
    
        
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


def load_batch_data(config, data_from_test_file, batch_size=200):
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ep_id_list = data_from_test_file['episode_id_list']
    df = data_from_test_file['total_data']
    label_column = config.get('general', 'label_column')
    #row_df = data_from_tune_file['row_df']
    #embedding_dict = data_from_tune_file['embedding_dict']
    #embedding_size = data_from_tune_file['embedding_size']
        
    #all_ep_ids = ep_id_list.to(device).numpy() #converting from torch
    #all_ep_ids = ep_id_list.cpu().numpy() #converting from torch
    
    
    #print("Total episodes: " + str(len(all_ep_ids)))
    #batch_size = len(all_ep_ids)
    batch_wise_ep_ids = np.array_split(ep_id_list, batch_size) # may be randomize it here later??
    
    #print(len(batch_wise_ep_ids))
    
    all_input_seq_list = []
    all_output_seq_list = []
    all_max_seq_length = []
    all_length_more_then_one_batch_wise_ep_ids =[]
    
    #exp_dir = '/Volumes/mahbub_veracrypt_mac/sample_test/'
    #label_file = open(exp_dir+'label_file_test.txt','a')
    
    for ep_ids in batch_wise_ep_ids:
        
        #print(len(ep_ids))
    
        input_seq_list = []
        output_seq_list = []
        length_more_then_one_ep_ids = []
        for ep_id in ep_ids:
            
            
        
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
            
            input_seq = seq.drop(columns=[label_column, 'icustay_id']).values.tolist()
            #print(input_seq)
            #print('done')
            
            '''
            #for embedding_value in embeddings_per_row:
                
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
            
                
            length_more_then_one_ep_ids.append(ep_id)
           
       
        #label_file.close()
        
        max_seq_length=max(list(map(len, output_seq_list)))
    
            #print('max_seq_length: '+str(max_seq_length))
    
            #min_seq_length = min(list(map(len, output_seq_list)))
            
            #print('min_seq_length: '+str(min_seq_length))
            
        #print(len(output_seq_list))
        
        if len(input_seq_list)==0:
            print ('Wronggggggggggg')
            
        all_input_seq_list.append(input_seq_list)
        all_output_seq_list.append(output_seq_list)
        all_max_seq_length.append(max_seq_length)
        all_length_more_then_one_batch_wise_ep_ids.append(np.array(length_more_then_one_ep_ids))
        
    '''
    print (len(all_input_seq_list))
    print (len(all_output_seq_list))
    print (len(all_max_seq_length))
    '''
    
    #if u use all then use batch_wise_ep_ids at last
    return all_input_seq_list, all_output_seq_list, all_max_seq_length, batch_wise_ep_ids
    #return all_input_seq_list, all_output_seq_list, all_output_string_seq_dict, all_max_seq_length, np.array(all_length_more_then_one_batch_wise_ep_ids)
