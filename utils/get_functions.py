import os
import sys

import torch
import torch.optim as optim

import numpy as np

def get_deivce() :
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("You are using \"{}\" device.".format(device))

    return device

def get_optimizer(args, model):
    params = list(filter(lambda p: p.requires_grad, model.parameters()))

    if args.optimizer_name == 'SGD' :
        optimizer = optim.SGD(params=params, lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)
    elif args.optimizer_name == 'Adam' :
        optimizer = optim.Adam(params=params, lr=args.lr)
    elif args.optimizer_name == 'AdamW' :
        optimizer = optim.AdamW(params=params, lr=args.lr, weight_decay=args.weight_decay)
    else :
        print("Wrong optimizer")
        sys.exit()

    return optimizer

def get_lr(step, total_steps, lr_max, lr_min):
  """Compute learning rate according to cosine annealing schedule."""
  return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

def get_scheduler(args, optimizer, train_loader_len) :
    if args.LRS_name == 'SLRS' : # step learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
    elif args.LRS_name == 'MSLRS': # multi-step learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    elif args.LRS_name == 'CALRS': # cosine learning rate scheduler
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
                step,
                args.final_epoch * train_loader_len,
                1,  # lr_lambda computes multiplicative factor
                1e-6 / args.lr))
    elif args.LRS_name == 'LinearScaledBatchSize': scheduler = None
    else: scheduler = None

    return scheduler

def get_save_path(args):
    save_model_path = '{}_{}x{}_{}_{}({}+{}_{})_{}({}_{})_{}'.format(args.train_data_type,
                                                           str(args.image_size), str(args.image_size),
                                                           str(args.train_batch_size),
                                                           args.model_name, args.cnn_backbone, args.transformer_backbone, args.model_scale,
                                                           args.optimizer_name,
                                                           args.lr,
                                                           str(args.final_epoch).zfill(3),
                                                           str(args.LRS_name))

    save_model_path = os.path.join(save_model_path, args.our_method_configuration)

    model_dirs = os.path.join(args.save_path, save_model_path)
    if not os.path.exists(os.path.join(model_dirs, 'model_weights')): os.makedirs(os.path.join(model_dirs, 'model_weights'))
    if not os.path.exists(os.path.join(model_dirs, 'test_reports')): os.makedirs(os.path.join(model_dirs, 'test_reports'))
    if not os.path.exists(os.path.join(model_dirs, 'plot_results')): os.makedirs(os.path.join(model_dirs, 'plot_results'))

    return model_dirs