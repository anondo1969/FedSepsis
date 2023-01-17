#!/usr/bin/env python
# coding: utf-8
'''
@author: Mahbub Ul Alam (mahbub@dsv.su.se)
@version: 1.0+
@copyright: Copyright (c) Mahbub Ul Alam (mahbub@dsv.su.se)
@license : MIT License
'''
import logging
import socket
import sys
from datetime import datetime
import torch
import time

import wandb

from lstm import RNN_LSTM
from lstm_trainer import LSTMTrainer
from fl_dataloader import local_dataloader, load_all_data
from managing_connection_socket import client_send_msg, client_recv_msg
from information_cpu import printPerformance
from evaluate_results import write_result_file_header


# python3 client_FedSepsis.py 1969 130.237.158.192

def create_model(args, device):
    data_from_train_file = load_all_data(args, args.test_or_tune)
    hidden_size = args.hidden_units
    num_layers = args.hidden_layers
    dropout = float(args.drop_out / 100.0)
    if num_layers == 1:
        dropout = 0
    num_classes = data_from_train_file['num_classes'] + 1
    input_size = data_from_train_file['input_size']

    model = RNN_LSTM(input_size, hidden_size, num_layers, num_classes, dropout).to(device)

    # logging.info(model)
    return model


# ---------------------------------------------------------------------------------------------------------------------------------------------------
print()
print("Experiment starts, current time:" +str(datetime.now()))
print()

start_time = time.time()  # store start time

port = int(sys.argv[1])
host = str(sys.argv[2])

client_server_socket = socket.socket()
client_server_socket.connect((host, port))

print()
print('Receiving instruction from the server....')
timestamp = time.time()

msg = client_recv_msg(client_server_socket)

print('Received, time taken: {:.0f}m {:.0f}s'.format((time.time() - timestamp) // 60, (time.time() - timestamp) % 60))

print()

args = msg['args']
client_id = msg['client_id']

write_result_file_header(client_id + 1, args)

wandb.init(

    project=args.experiment_name+'_'+args.experiment_category+'_total_clients_'+str(args.client_num_in_total)+'_average_choice_'+args.average_choice, entity='mahbub',
    name="client_" + str(client_id + 1),
    config=args
)

wandb.log({"Temperature": printPerformance()})

print('epochs = ' + str(args.epochs))
print('rounds = ' + str(args.rounds))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print()
print("creating model")
model = create_model(args, device)
print("model creation done")
print()

print("creating data")
train_data_local, train_data_local_num = local_dataloader(args, 'train', client_id)
test_data, test_data_num = local_dataloader(args, args.test_or_tune, client_id)
print("data creation done")
print()
print('Sending the total number of classification data to the server....')
timestamp = time.time()

client_send_msg(client_server_socket, train_data_local_num)

print('Sent, time taken: {:.0f}m {:.0f}s'.format((time.time() - timestamp) // 60, (time.time() - timestamp) % 60))

for r in range(args.rounds):  # loop over the dataset multiple times

    # msg=[]

    print()
    print('Rounds {}/{}'.format(r + 1, args.rounds))
    print('=' * 10)
    print()

    print('Receiving average weights from the server....')
    timestamp = time.time()

    server_model_weights = client_recv_msg(client_server_socket)

    print(
        'Received, time taken: {:.0f}m {:.0f}s'.format((time.time() - timestamp) // 60, (time.time() - timestamp) % 60))
    print("Current time:" +str(datetime.now()))
    print()

    wandb.log({"Temperature": printPerformance()})

    print("Training starts.")

    model_trainer = LSTMTrainer(model)
    model_trainer.set_model_params(server_model_weights)
    model_trainer.train(train_data_local, device, args)
    print("Training ends.")
    print()
    
    results_list = model_trainer.test(test_data, device, args, r + 1, client_id+1)
    client_model_weights = model_trainer.get_model_params()
    

    wandb.log({"Temperature": printPerformance()})

    print()

    print('Sending locally trained model weights to the server....')

    timestamp = time.time()
    client_model_weights_and_results = [client_model_weights, results_list]
    client_send_msg(client_server_socket, client_model_weights_and_results)

    print('Sent, time taken: {:.0f}m {:.0f}s'.format((time.time() - timestamp) // 60, (time.time() - timestamp) % 60))
    print()
    print("Current time:" +str(datetime.now()))
    print()

print()
print('Finished Training, Total time taken: {:.0f}m {:.0f}s'.format((time.time() - start_time) // 60,
                                                                    (time.time() - start_time) % 60))

