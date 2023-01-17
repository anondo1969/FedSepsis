'''
@author: Mahbub Ul Alam (mahbub@dsv.su.se)
@version: 1.0+
@copyright: Copyright (c) Mahbub Ul Alam (mahbub@dsv.su.se)
@license : MIT License
'''
import logging

import torch
import torch.nn as nn
import wandb
import os
from torch import LongTensor
from model_trainer import ModelTrainer
import fl_dataloader
import evaluate_results
#from jtop import jtop

class LSTMTrainer(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):

        self.args = args

        data_from_train_file = fl_dataloader.load_all_data(args, 'train')

        whole_data = data_from_train_file['total_data']
        row_df = data_from_train_file['row_df']
        embedding_dict = data_from_train_file['embedding_dict']
        embedding_size = data_from_train_file['embedding_size']

        num_classes = data_from_train_file['num_classes'] + 1
        input_size = data_from_train_file['input_size']

        num_epochs = args.epochs

        # logging.info(device)

        model = self.model.to(device)
        model.train()

        criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=args.pad_integer)

        optimizer = torch.optim.Adam(model.parameters())

        for epoch in range(num_epochs):

            for batch_count, (batch_wise_episode_id_list, target) in enumerate(train_data):

                # take JetsonNano stats, only provide in jetson nano clients

                #if device=='cuda':
                    #with jtop() as jetson:
                        #stats = jetson.stats
                        #print("")
                        #print(stats)
                        #print("")
                        #wandb.log(stats)

                # print ("batch index {}, 0/1: {}/{}".format(
                # batch_count,
                # len(np.where(target.numpy() == 0)[0]),
                # len(np.where(target.numpy() == 1)[0])))

                # logging.info("Batch = %d" % (batch_count+1))

                # implement here

                if args.experiment_category == 'non-text':

                    X_list, Y_list, max_seq_length = fl_dataloader.load_batch_data_non_text(batch_wise_episode_id_list,
                                                                                            whole_data,
                                                                                            args.label_column,
                                                                                            args.all_pos_all_neg)

                elif args.experiment_category == 'text':

                    X_list, Y_list, max_seq_length = fl_dataloader.load_batch_data_text(batch_wise_episode_id_list,
                                                                                        whole_data, row_df,
                                                                                        embedding_dict, embedding_size,
                                                                                        args.label_column,
                                                                                        args.all_pos_all_neg)

                elif args.experiment_category == 'multi-modal':

                    X_list, Y_list, max_seq_length = fl_dataloader.load_batch_data_multi_modal(
                        batch_wise_episode_id_list, whole_data, row_df, embedding_dict, embedding_size,
                        args.label_column,
                        args.all_pos_all_neg)
                '''
                X_list, Y_list, max_seq_length = load_batch_data(batch_wise_episode_id_list,
                                                                 whole_data, row_df,
                                                                 embedding_dict,
                                                                 embedding_size,
                                                                 args.label_column,
                                                                 args.all_pos_all_neg)
                '''

                # print('max_seq_length: '+str(max_seq_length))

                seq_lengths = LongTensor(list(map(len, X_list))).to(device)
                # print('seq_lengths: '+ str(seq_lengths))
                # change here again double and float
                seq_tensor = torch.Tensor(args.batch_size, max_seq_length, input_size).float().to(device)
                seq_tensor = seq_tensor.fill_(args.pad_integer)

                for idx, (seq, seqlen) in enumerate(zip(X_list, seq_lengths)):
                    # print (seq)
                    # seq_tensor[idx, :seqlen] = FloatTensor(seq)
                    seq_tensor[idx, :seqlen] = seq

                # print('seq_tensor.shape: '+str(seq_tensor.shape))

                label_tensor = torch.Tensor(args.batch_size, max_seq_length).long().to(device)
                label_tensor = label_tensor.fill_(args.pad_integer)
                for idx, (seq, seqlen) in enumerate(zip(Y_list, seq_lengths)):
                    # label_tensor[idx, :seqlen] = LongTensor(seq)
                    label_tensor[idx, :seqlen] = seq

                # print('label_tensor.shape: '+str(label_tensor.shape))

                seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
                seq_tensor = seq_tensor[perm_idx]
                label_tensor = label_tensor[perm_idx]

                # Forward pass
                outputs = model(seq_tensor, seq_lengths)

                outputs = outputs.view(-1, num_classes)

                # print('final_outputs.shape: '+str(outputs.shape))

                labels = label_tensor.view(-1)

                # print('labels.shape: '+str(labels.shape))

                loss = criterion(outputs, labels)

                wandb.log({"loss": loss})
                # logging.info("Loss = %f" % (loss))

                if (batch_count + 1) % 5 == 0:
                    # logging.info("Batch = %d/%d, Loss = %0.3f" % (batch_count + 1, len(train_data), loss))
                    print("Batch = %d/%d, Loss = %0.3f" % (batch_count + 1, len(train_data), loss))
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # if batch_count+1 == 5:
                # print("Breaking just to check")
                # break

    def test_local(self, test_data, device, args):
        pass

    def test(self, test_data, device, args, round, client_id):  # -1 means server

        print("Testing starts.")
        model = self.model.to(device)
        self.model.eval()

        self.args = args

        data_from_test_file = fl_dataloader.load_all_data(args, args.test_or_tune)

        pad_integer = 0

        input_size = data_from_test_file['input_size']
        num_classes = data_from_test_file['num_classes'] + 1  # for the dummy class in padding (0)
        whole_data = data_from_test_file['total_data']
        row_df = data_from_test_file['row_df']
        embedding_dict = data_from_test_file['embedding_dict']
        embedding_size = data_from_test_file['embedding_size']

        Y_true = []
        Y_predicted = []
        Y_predicted_score = []
        Episode_id_list_sorted = []
        Seq_lengths_sorted = []

        for batch_count, (batch_wise_episode_id_list, target) in enumerate(test_data):

            if args.experiment_category == 'non-text':

                X_list, Y_list, max_seq_length = fl_dataloader.load_batch_data_non_text(batch_wise_episode_id_list,
                                                                                        whole_data, args.label_column,
                                                                                        0)  # args.all_pos_all_neg = 0 because we can not change test data labels

            elif args.experiment_category == 'text':

                X_list, Y_list, max_seq_length = fl_dataloader.load_batch_data_text(batch_wise_episode_id_list,
                                                                                    whole_data, row_df, embedding_dict,
                                                                                    embedding_size, args.label_column,
                                                                                    0)  # args.all_pos_all_neg = 0 because we can not change test data labels

            elif args.experiment_category == 'multi-modal':

                X_list, Y_list, max_seq_length = fl_dataloader.load_batch_data_multi_modal(batch_wise_episode_id_list,
                                                                                           whole_data, row_df,
                                                                                           embedding_dict,
                                                                                           embedding_size,
                                                                                           args.label_column,
                                                                                           0)  # args.all_pos_all_neg = 0 because we can not change test data labels
            '''
            X_list, Y_list, max_seq_length = load_batch_data(batch_wise_episode_id_list,
                                                             whole_data, row_df,
                                                             embedding_dict,
                                                             embedding_size,
                                                             args.label_column,
                                                             0)  # args.all_pos_all_neg = 0 because we can not change test data labels
            '''

            # print('max_seq_length: '+str(max_seq_length))

            seq_lengths = LongTensor(list(map(len, X_list))).to(device)
            # print('seq_lengths: '+ str(seq_lengths))
            # change here again double and float
            seq_tensor = torch.Tensor(args.test_batch_size, max_seq_length, input_size).float().to(device)
            seq_tensor = seq_tensor.fill_(args.pad_integer)

            for idx, (seq, seqlen) in enumerate(zip(X_list, seq_lengths)):
                # print (seq)
                # seq_tensor[idx, :seqlen] = FloatTensor(seq)
                seq_tensor[idx, :seqlen] = seq

            # print('seq_tensor.shape: '+str(seq_tensor.shape))

            label_tensor = torch.Tensor(args.test_batch_size, max_seq_length).long().to(device)
            label_tensor = label_tensor.fill_(args.pad_integer)
            for idx, (seq, seqlen) in enumerate(zip(Y_list, seq_lengths)):
                # label_tensor[idx, :seqlen] = LongTensor(seq)
                label_tensor[idx, :seqlen] = seq

            # print('label_tensor.shape: '+str(label_tensor.shape))

            seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
            seq_tensor = seq_tensor[perm_idx]
            label_tensor = label_tensor[perm_idx]
            episode_id_list_sorted = batch_wise_episode_id_list[perm_idx.cpu()]
            # Forward pass
            outputs = model(seq_tensor, seq_lengths)

            outputs = torch.exp(outputs)

            outputs = outputs.view(-1, num_classes)

            labels = label_tensor.view(-1)

            y_true = labels.to(device).tolist()

            predicted_prob, y_predicted = torch.max(outputs.data, 1)

            y_predicted = y_predicted.to(device).tolist()
            predicted_prob = predicted_prob.to(device).tolist()

            predicted_score = []
            outputs_list = outputs.to(device).tolist()

            for timestep in range(len(outputs_list)):
                if sum(outputs_list[timestep]) != 0:
                    predicted_score.append((outputs_list[timestep][2]) / sum(outputs_list[timestep]))
                else:
                    print('predicted score is zero')
                    predicted_score.append(0)

            for value_index in range(len(y_true)):
                Y_true.append(y_true[value_index])
                Y_predicted.append(y_predicted[value_index])
                Y_predicted_score.append(predicted_score[value_index])

            seq_lengths = seq_lengths.to(device).tolist()

            for value_index in range(len(seq_lengths)):
                Episode_id_list_sorted.append(episode_id_list_sorted[value_index])
                Seq_lengths_sorted.append(seq_lengths[value_index])

            # print('Batch '+str(index+1)+' is completed')

        decoding_details = {'Y_true': Y_true,
                            'Y_predicted': Y_predicted,
                            'Y_predicted_score': Y_predicted_score,
                            'episode_id_list_sorted': Episode_id_list_sorted,
                            'seq_lengths_sorted': Seq_lengths_sorted
                            }

        results_list = self.evaluation(decoding_details, args, client_id, round)

        print("Testing ends.")
        print()

        return results_list

    def evaluation(self, decoding_details, args, client_id, round):

        # print('\nEpisode wise results: ')

        timestep_wise_y_true_and_y_predicts = evaluate_results.remove_dummy_pad_value_zero_from_timestep_wise_y_true_and_y_predict(
            decoding_details)
        results_list = []
        wandb_evaluation_names = ["episode_wise_accuracy", "episode_wise_true_positives",
                                  "episode_wise_true_negatives",
                                  "episode_wise_false_positives", "episode_wise_false_negatives",
                                  "episode_wise_recall",
                                  "episode_wise_precision", "episode_wise_f1_score", "episode_wise_AU_ROC",
                                  "episode_wise_AU_PRC", "earliness"]
        wandb_evaluation_values = []
        for settings in range(1, 4):
            for range_limit in [0, 24, 48]:
                episode_wise_evaluation_scores = evaluate_results.compute_episode_wise_evaluation_scores(
                    timestep_wise_y_true_and_y_predicts,
                    decoding_details, settings, range_limit, args, args.test_or_tune)

                earliness = evaluate_results.compute_episodes_wise_earliness(timestep_wise_y_true_and_y_predicts,
                                                                             decoding_details, settings, range_limit,
                                                                             args, args.test_or_tune)

                results_line = evaluate_results.write_experiment_information(earliness, episode_wise_evaluation_scores,
                                                                             settings,
                                                                             range_limit, args, round, client_id)

                results_list.append(results_line)

                # for wandb logs, graphs
                if settings == 1 and range_limit == 0:
                    for episode_wise_evaluation_score in episode_wise_evaluation_scores:
                        wandb_evaluation_values.append(episode_wise_evaluation_score)

                    wandb_evaluation_values.append(earliness)

                for index, evaluation_value in enumerate(wandb_evaluation_values):
                    wandb.log({wandb_evaluation_names[index] + '_after_round_' + str(round): evaluation_value})

                '''

                print('settings: ' + str(settings) + ' range_limit: ' + str(range_limit))
                print('accuracy\ttp\ttn\tfp\tfn\trecall\tprecision\tf1_score\tAU_ROC\tAU_PRC')
                print()
                print(episode_wise_evaluation_scores)
                print()
                print('earliness: ' + str(earliness))
                print()

                '''

        return results_list

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None):
        

        model_save_dir = "../../training"
        path = os.path.join(model_save_dir, 'model.ckpt')
        torch.save(self.model.state_dict(), path)

        return True
