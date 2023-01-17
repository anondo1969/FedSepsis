'''
@author: Mahbub Ul Alam (mahbub@dsv.su.se)
@version: 1.0+
@copyright: Copyright (c) Mahbub Ul Alam (mahbub@dsv.su.se)
@license : MIT License
'''
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
import pandas as pd
import pickle
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

clinical_bert = sys.argv[1]
start_count = sys.argv[2]

#change model name here

if clinical_bert=='huang':
    tokenizer = AutoTokenizer.from_pretrained("anondo/test_anon")
    model = AutoModel.from_pretrained("anondo/test_anon").to(device)
    config = AutoConfig.from_pretrained("anondo/test_anon", output_hidden_states=True, hidden_size=768)

else:

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)
    config = AutoConfig.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", output_hidden_states=True, hidden_size=768)

#model.to(device)
model.eval()

with open('non_text_row_id_mapping_text_row_id_dict.pkl', 'rb') as f:
    non_text_row_id_mapping_text_row_id_dict = pickle.load(f)
    
sepsis_text = pd.read_csv('sepsis_text.csv')

token_vecs_sum = []
token_vecs_cat = []
long_embedding_dict = dict()
short_embedding_dict = dict()
long_embedding_dict_last = dict()
short_embedding_dict_last = dict()
count=0
total = len(non_text_row_id_mapping_text_row_id_dict)

for index in non_text_row_id_mapping_text_row_id_dict:
    
    count+=1
    
    if count >= int(start_count):
    
        text_row_ids = non_text_row_id_mapping_text_row_id_dict[index][0]
        row_text_list=sepsis_text.loc[sepsis_text['ROW_ID'].isin(text_row_ids)]['TEXT'].tolist()
    
        if len(row_text_list) == 0:
        
            long_sentence_embedding=torch.zeros(3072).to(device)
            short_sentence_embedding=torch.zeros(768).to(device)
        
        else:
        
            batch = tokenizer(row_text_list, padding=True, truncation=True, return_tensors="pt", max_length = 512).to(device)
        
            with torch.no_grad():
            
                outputs = model(batch['input_ids'], batch['token_type_ids'], output_hidden_states=True)
            
            hidden_states= outputs.hidden_states
        
            token_embeddings = torch.stack(hidden_states, dim=0)
        
            row_token_embeddings = torch.mean(token_embeddings, 1, True)
        
            token_embeddings = torch.squeeze(row_token_embeddings, dim=1)
        
            token_embeddings = token_embeddings.permute(1,0,2)
        
            # Stores the token vectors, with shape [512 x 768]
            token_vecs_sum = []
        
            # Stores the token vectors, with shape [512x 3,072]
            token_vecs_cat = []

            # `token_embeddings` is a [512x 12 x 768] tensor.

            # For each token in the sentence...
            for token in token_embeddings:

                # `token` is a [12 x 768] tensor

                # Sum the vectors from the last four layers.
                sum_vec = torch.sum(token[-4:], dim=0)
    
                # Use `sum_vec` to represent `token`.
                token_vecs_sum.append(sum_vec)
        
                # Concatenate the vectors (that is, append them together) from the last 
                # four layers.
                # Each layer vector is 768 values, so `cat_vec` is length 3,072.
                cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
    
                # Use `cat_vec` to represent `token`.
                token_vecs_cat.append(cat_vec)

                #print ('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))
                #print ('Shape is: %d x %d' % (len(token_vecs_cat), len(token_vecs_cat[0])))
        
                long_sentence_embedding = torch.mean(torch.stack(token_vecs_cat), dim=0)
                short_sentence_embedding = torch.mean(torch.stack(token_vecs_sum), dim=0)
        
    
        long_embedding_dict[index] = long_sentence_embedding
        short_embedding_dict[index] = short_sentence_embedding
    
    
    
        print(str(count)+'/'+str(total)+' is done.')
    
        if count%10000==0:
    
    	    torch.save(long_embedding_dict, clinical_bert+'_torch_long_embedding_dict_'+str(count))
    	    torch.save(short_embedding_dict, clinical_bert+'_torch_short_embedding_dict_'+str(count))
    	
    	    print('Saved torch_long_embedding_dict_'+str(count))
    	
    	    long_embedding_dict.clear()
    	    short_embedding_dict.clear()
    	
        elif count == len(non_text_row_id_mapping_text_row_id_dict):
    
            torch.save(long_embedding_dict, clinical_bert+'_torch_long_embedding_dict_last')
            torch.save(short_embedding_dict, clinical_bert+'_torch_short_embedding_dict_last')
    	
            print('Saved torch_long_embedding_dict_last')
    	
