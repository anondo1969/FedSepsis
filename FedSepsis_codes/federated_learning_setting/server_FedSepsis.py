#!/usr/bin/env python
# coding: utf-8


import argparse
import logging
import socket
import torch
from threading import Thread
from threading import Lock
import time
import copy

import wandb

from managing_connection_socket import server_send_msg, server_recv_msg
from lstm import RNN_LSTM
from federated_algorithms import compute_federated_algorithm
from lstm_trainer import LSTMTrainer
from fl_dataloader import local_dataloader, load_all_data
from evaluate_results import write_result_file_header, write_client_test_results_on_server



# python3 server_FedSepsis.py --client_num_in_total 2 --experiment_name nano_opt_2 --port 1969 --rounds 4 --epochs 1 --host 130.237.158.192 --average_choice opt

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    parser.add_argument('--experiment_name', type=str,
                        help='experiment name, jetson_nano or raspberry_pi')

    parser.add_argument('--port', type=int,
                        help='port address of the FedML server')

    parser.add_argument('--data_dir', type=str, default='../data/SEPSIS',
                        help='data directory')

    parser.add_argument('--exp_dir', type=str, default='../exp',
                        help='experiment directory')

    parser.add_argument('--client_num_in_total', type=int, default=1, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                        help='input batch size for training (default: 50)')

    parser.add_argument('--test_batch_size', type=int, default=50, metavar='N',
                        help='input batch size for testing (default: 50)')

    parser.add_argument('--epochs', type=int, default=2, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--rounds', type=int, default=2,
                        help='how many round of communications we should use')

    parser.add_argument('--host', type=str, default="10.200.61.211",
                        help='IP address of the FedML server')

    parser.add_argument('--imputation', type=str, default="imputed",
                        help='missing data imputation method')

    parser.add_argument('--time_window', type=int, default=1,
                        help='hourly bin')

    parser.add_argument('--hidden_units', type=int, default=800,
                        help='hidden units')

    parser.add_argument('--hidden_layers', type=int, default=2,
                        help='hidden layers')

    parser.add_argument('--drop_out', type=int, default=20,
                        help='drop out value')

    parser.add_argument('--clinicalBERT', type=str, default="Emily",
                        help='clinicalBERT')

    parser.add_argument('--embedding_type', type=str, default="short",
                        help='embedding_type')

    parser.add_argument('--pad_integer', type=int, default=0,
                        help='pad_integer')

    parser.add_argument('--all_pos_all_neg', type=int, default=1,
                        help='all_pos_all_neg')

    parser.add_argument('--change', type=str, default='with_train_label_change',
                        help='with_train_label_change')

    parser.add_argument('--label_column', type=str, default="sepsis_target",
                        help='label_column')

    parser.add_argument('--average_choice', type=str, default="simple",
                        help='federated algorithm')

    parser.add_argument('--test_or_tune', type=str, default="tune",
                        help='test data or tune data')

    parser.add_argument('--server_optimizer', type=str, default='adam',
                        help='Optimizer used on the server. This field can be the name of any subclass of the torch Opimizer class.')

    parser.add_argument('--server_lr', type=float, default=0.001,
                        help='server learning rate (default: 0.001)')

    parser.add_argument('--experiment_category', type=str,
                        help='experiment_category, multi-modal, text, or non-text (default: none, you must give it)')


    args = parser.parse_args()
    return args


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


def run_thread(func, model, args, s):
    global clientsoclist
    global start_time

    thrs = []
    for i in range(args.client_num_in_total):
        conn, addr = s.accept()
        print('Connected with client ' + str(i + 1) + ' : ', addr)
        # append client socket on list
        clientsoclist[i] = conn
        # args = (i, num_user, conn)
        # thread = Thread(target=func, args=args)
        # thrs.append(thread)
        # thread.start()
        thrs.append(Thread(target=func, args=(i, model, args, conn)))

    print()
    print("All clients have connected! The process starts now!")
    print()

    start_time = time.time()  # store start time

    for thread in thrs:
        thread.start()

    for thread in thrs:
        thread.join()
    # end_time = time.time()  # store end time
    # print("TrainingTime: {} sec".format(end_time - start_time))


def receive(userid, model, args, conn):  # thread for receive clients
    global weight_count

    global datasetsize

    # change here
    msg = dict()
    msg['client_id'] = userid
    msg['args'] = args

    datasize = server_send_msg(conn, msg)  # send epoch

    print("Sent info about round, id, and epoch to client " + str(userid + 1))

    total_sendsize_list.append(datasize)
    client_sendsize_list[userid].append(datasize)

    train_dataset_size, datasize = server_recv_msg(conn)  # get length of train dataset
    print()
    print("received info about dataset_size and train dataset size from client " + str(userid + 1) + ", " + str(
        datasize) + ", " + str(train_dataset_size))
    print()
    total_receivesize_list.append(datasize)
    client_receivesize_list[userid].append(datasize)

    with lock:
        datasetsize[userid] = train_dataset_size
        weight_count += 1

    print("weight_count: " + str(weight_count))
    print()
    train(userid, model, args, conn)


def train(userid, model, args, client_conn):
    global weights_list
    global performance_metrics
    global global_weights
    global weight_count

    for round in range(args.rounds):
        with lock:
            if weight_count == args.client_num_in_total:

                # torch.save(global_weights, "exp/model_round_" + str(round) + ".pth")
                # print("Saved average weight after round " + str(round))

                # do code for evaluation
                # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                # model = create_model(args, device)
                # model_trainer = LSTMTrainer(model)
                # model_trainer.set_model_params(global_weights)
                # test_data, test_data_num = local_dataloader(args, args.test_or_tune, -1)
                # _ = model_trainer.test(test_data, device, args, round + 1, -1)

                for client_id, conn in enumerate(clientsoclist):
                    # if round==0:

                    for key in global_weights.keys():
                        global_weights[key] = global_weights[key].cpu()
                    # print(global_weights)
                    datasize = server_send_msg(conn, global_weights)
                    print("sent average weight to client " + str(client_id + 1))

                    total_sendsize_list.append(datasize)
                    client_sendsize_list[client_id].append(datasize)
                    train_sendsize_list.append(datasize)
                    weight_count = 0

                print()

        print("waiting to receive model weight from client " + str(userid + 1) + "....")
        print()
        # client_model_weights_and_results = [client_model_weights, results_list]
        # client_weights, datasize = server_recv_msg(client_conn)
        client_model_weights_and_results, datasize = server_recv_msg(client_conn)
        print("received model weights and results from client " + str(userid + 1))
        print()
        total_receivesize_list.append(datasize)
        client_receivesize_list[userid].append(datasize)
        train_receivesize_list.append(datasize)

        # client_model_weights_and_results = [client_model_weights, results_list]
        client_weights = client_model_weights_and_results[0]
        results = client_model_weights_and_results[1]
        write_client_test_results_on_server(args, results=results, header=False)
        weights_list[userid] = client_weights

        print("Client " + str(userid + 1) + " : Round " + str(round + 1) + " : done")
        print()
        with lock:
            weight_count += 1
            if weight_count == args.client_num_in_total:
                # average
                print()
                print("Federated algorithm computation starts")
                model, global_weights = compute_federated_algorithm(weights_list, model, args, round)
                print("Federated algorithm computation ends")
                print()

                model_trainer = LSTMTrainer(model)
                test_data, test_data_num = local_dataloader(args, args.test_or_tune, -1)
                _ = model_trainer.test(test_data, device, args, round + 1, -1)

# -----------------------------------------------------------------------------
# parser = argparse.ArgumentParser()
# args = add_args(parser)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# global_weights = torch.load("exp/model_round_1.pth", map_location=device )
# model = create_model(args, device)
# model_trainer = LSTMTrainer(model)
# model_trainer.set_model_params(global_weights)
# test_data, test_data_num = local_dataloader(args, args.test_or_tune, -1)
# print("test_data_num "+str(test_data_num))
# results_list = model_trainer.test(test_data, device, args, 1, -1)

# '''
start_time = time.time()

parser = argparse.ArgumentParser()
args = add_args(parser)

write_result_file_header(-1, args)
write_client_test_results_on_server(args, results=None, header=True)

wandb.init(

    project=args.experiment_name+'_'+args.experiment_category+'_total_clients_'+str(args.client_num_in_total)+'_average_choice_'+args.average_choice, entity='mahbub',
    name="server",
    config=args
)

print()
print('Waiting for all ' + str(args.client_num_in_total) + ' clients to join, after all ' + str(
    args.client_num_in_total) + ' clients are joined, we will start...')
print()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("creating model")
model = create_model(args, device)
print("model creation done")
print()

clientsoclist = [0] * args.client_num_in_total

start_time = 0
weight_count = 0

global_weights = copy.deepcopy(model.state_dict())

datasetsize = [0] * args.client_num_in_total
weights_list = [0] * args.client_num_in_total
performance_metrics = [0] * args.client_num_in_total

lock = Lock()

total_sendsize_list = []
total_receivesize_list = []

client_sendsize_list = [[] for i in range(args.client_num_in_total)]
client_receivesize_list = [[] for i in range(args.client_num_in_total)]

train_sendsize_list = []
train_receivesize_list = []

# host = socket.gethostbyname(socket.gethostname())
host = socket.gethostbyname("")
# print(host)


# s = socket.socket()
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((args.host, args.port))
server_socket.listen(128)

# s.bind((TCP_IP, TCP_PORT))


run_thread(receive, model, args, server_socket)

checkpoint_path = args.exp_dir + "/" + args.experiment_name+'_'+args.experiment_category+'_total_clients_'+str(args.client_num_in_total)+'_average_choice_'+args.average_choice + ".pth"

torch.save(global_weights, checkpoint_path)

# Final testing
model_trainer = LSTMTrainer(model)
model_trainer.set_model_params(global_weights)
test_data, test_data_num = local_dataloader(args, args.test_or_tune, -1)
_ = model_trainer.test(test_data, device, args, args.rounds + 1, -1)

print('Finished training, Total time taken: {:.0f}m {:.0f}s'.format((time.time() - start_time) // 60,
                                                                    (time.time() - start_time) % 60))
print()
print('total_sendsize_list')
print(total_sendsize_list)

print('total_receivesize_list')
print(total_receivesize_list)

print('client_sendsize_list')
print(client_sendsize_list)

print('client_receivesize_list')
print(client_receivesize_list)

print('train_sendsize_list')
print(train_sendsize_list)

print('train_receivesize_list')
print(train_receivesize_list)
# '''
