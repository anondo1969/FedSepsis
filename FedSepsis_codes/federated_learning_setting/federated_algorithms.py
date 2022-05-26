#!/usr/bin/env python
# coding: utf-8

import copy

import numpy as np
import torch
from optrepo import OptRepo


def compute_federated_algorithm(weights_list, model, args, round):
    if args.average_choice == 'simple':

        print("\nAlgorithm name: Fedavg\nPaper name: Communication Efﬁcient Learning of Deep Networks from Decentralized Data\n")

        global_weights = average_weights_simple(weights_list)
        model.load_state_dict(global_weights)

    elif args.average_choice == 'opt':

        print("\nAlgorithm name: Fedopt\nPaper name: Adaptive Federated Optimization\n")

        if round == 0:  # no point fine-tuning an empty weights
            global_weights = average_weights_simple(weights_list)
            model.load_state_dict(global_weights)

        else:
            model, global_weights = average_weights_opt(weights_list, model, args)

    return model, global_weights


def average_weights(w, datasize):
    """
    Returns the average of the weights.
    """

    for i, data in enumerate(datasize):
        for key in w[i].keys():
            w[i][key] *= float(data)

    w_avg = copy.deepcopy(w[0])

    # when client use only one kind of device

    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], float(sum(datasize)))

    # when client use various devices (cpu, gpu) you need to use it instead
    #
    #     for key, val in w_avg.items():
    #         common_device = val.device
    #         break
    #     for key in w_avg.keys():
    #         for i in range(1, len(w)):
    #             if common_device == 'cpu':
    #                 w_avg[key] += w[i][key].cpu()
    #             else:
    #                 w_avg[key] += w[i][key].cuda()
    #         w_avg[key] = torch.div(w_avg[key], float(sum(datasize)))

    return w_avg


def average_weights_opt(w, model, args):
    w_avg = average_weights_simple(w)

    opt = instantiate_opt(args, model)

    opt.zero_grad()
    opt_state = opt.state_dict()
    # ----------------------------

    new_model = copy.deepcopy(model)
    new_model.load_state_dict(w_avg)
    with torch.no_grad():
        for parameter, new_parameter in zip(
                model.parameters(), new_model.parameters()
        ):
            parameter.grad = parameter.data - new_parameter.data
            # because we go to the opposite direction of the gradient
    model_state_dict = model.state_dict()
    new_model_state_dict = new_model.state_dict()
    for k in dict(model.named_parameters()).keys():
        new_model_state_dict[k] = model_state_dict[k]

    model.load_state_dict(new_model_state_dict)

    # ----------------------------
    opt = instantiate_opt(args, model)
    opt.load_state_dict(opt_state)
    opt.step()

    return model, new_model_state_dict


def instantiate_opt(args, model):
    opt = OptRepo.name2cls(args.server_optimizer)(
        # self.model_global.parameters(), lr=self.args.server_lr
        model.parameters(), lr=args.server_lr,
        # momentum=0.9 # for fedavgm
        # eps = 1e-3 for adaptive optimizer
    )

    return opt


def average_weights_simple(w):
    """
    Returns the average of the weights.
    """

    w_avg = copy.deepcopy(w[0])

    # when client use only one kind of device

    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]

        w_avg[key] = torch.div(w_avg[key], len(w))

    # when client use various devices (cpu, gpu) you need to use it instead
    #
    #     for key, val in w_avg.items():
    #         common_device = val.device
    #         break
    #     for key in w_avg.keys():
    #         for i in range(1, len(w)):
    #             if common_device == 'cpu':
    #                 w_avg[key] += w[i][key].cpu()
    #             else:
    #                 w_avg[key] += w[i][key].cuda()
    #         w_avg[key] = torch.div(w_avg[key], len(w))

    return w_avg


def average_weights_std_dev(w, metrics, metric_type):
    """
    if the metric is greater than (average of metrics - std dev), then
            the weights are used for averaging, else the weights are discarded.
    """

    metrics = np.asarray(metrics)
    std_dev = metrics.std()
    avg = metrics.mean()

    selected_weights = []

    for i, weight in enumerate(w):

        curr_metric = metrics[i]

        if metric_type == "loss":
            criteria = curr_metric <= (avg + std_dev)
        else:
            criteria = curr_metric >= (avg - std_dev)

        if criteria:
            selected_weights.append(weight)

        else:
            if metric_type == "loss":
                print(f"Client {i}: {curr_metric} > {avg + std_dev}")
            else:
                print(f"Client {i}: {curr_metric} < {avg - std_dev}")

    w_avg = copy.deepcopy(selected_weights[0])

    # when client use only one kind of device

    for key in w_avg.keys():
        for i in range(1, len(selected_weights)):
            w_avg[key] += selected_weights[i][key]
        w_avg[key] = torch.div(w_avg[key], len(selected_weights))

    return w_avg
