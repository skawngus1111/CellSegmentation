import os

import torch

import numpy as np
import pandas as pd

from utils.get_functions import get_save_path

def save_result(args, model, optimizer, history, test_results):
    model_dirs = get_save_path(args)

    print("Your experiment is saved in {}.".format(model_dirs))

    print("STEP1. Save {} Model Weight...".format(args.model_name))
    save_model(args, model, optimizer, model_dirs)

    print("STEP2. Save {} Model Test Results...".format(args.model_name))
    save_metrics(args, test_results, model_dirs)

    print("STEP3. Save {} Model History...".format(args.model_name))
    save_loss(history, model_dirs)

    print("EPOCH {} model is successfully saved at {}".format(args.final_epoch, model_dirs))

def save_model(args, model, optimizer, model_dirs):
    check_point = {
        'model_state_dict': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'current_epoch': args.final_epoch
    }

    torch.save(check_point, os.path.join(model_dirs, 'model_weights/model_weight(EPOCH {})_Trial{}.pth.tar'.format(args.final_epoch, args.current_trial)))

def save_metrics(args, test_results, model_dirs):
    print("###################### TEST REPORT ######################")
    for metric in test_results.keys():
        print("Mean {}    :\t {}".format(metric, test_results[metric]))
    print("###################### TEST REPORT ######################\n")

    if args.train_data_type == args.test_data_type:
        test_results_save_path = os.path.join(model_dirs, 'test_reports','test_report(EPOCH {})_Trial{}.txt'.format(args.final_epoch, args.current_trial))
    else:
        test_results_save_path = os.path.join(model_dirs, 'test_reports', 'Generalizability test_reports(EPOCH {})({}->{})_Trial{}.txt'.format(args.final_epoch, args.train_data_type, args.test_data_type, args.current_trial))

    f = open(test_results_save_path, 'w')

    f.write("###################### TEST REPORT ######################\n")
    for metric in test_results.keys():
        f.write("Mean {}    :\t {}\n".format(metric, test_results[metric]))
    f.write("###################### TEST REPORT ######################\n")

    f.close()

    print("test results txt file is saved at {}".format(test_results_save_path))

def save_loss(history, model_dirs):
    pd.DataFrame(history).to_csv(os.path.join(model_dirs, 'loss.csv'), index=False)

def save_total_trial_result(args):
    model_dirs = get_save_path(args)

    total_metrics_dict = dict()

    for metric in args.metric_list:
        total_metrics_dict[metric] = list()

    for current_trial in range(1, 4):
        print("Loading {} Trial results...".format(current_trial))

        if args.train_data_type == args.test_data_type:
            load_results_file = os.path.join(model_dirs, 'test_reports', 'test_report(EPOCH {})_Trial{}.txt'.format(args.final_epoch, current_trial))
        else:
            load_results_file = os.path.join(model_dirs, 'test_reports', 'Generalizability test_reports(EPOCH {})({}->{})_Trial{}.txt'.format(args.final_epoch, args.train_data_type, args.test_data_type, current_trial))

        f = open(load_results_file)
        while True:
            line = f.readline()
            if not line: break

            if line.split()[1] in args.metric_list: total_metrics_dict[line.split()[1]].append(float(line.split()[-1]))

        f.close()

    print("###################### TEST REPORT ######################")
    for metric in total_metrics_dict.keys():
        print("Trial Mean | Medium {}   :\t {} | {} ({}) [{} | {} | {}]".format(metric,
                                                                   np.round(np.mean(total_metrics_dict[metric]), 4),
                                                                   np.round(np.median(total_metrics_dict[metric]), 4),
                                                                   np.round(np.std(total_metrics_dict[metric]), 4),
                                                                   np.round(total_metrics_dict[metric][0], 4),
                                                                   np.round(total_metrics_dict[metric][1], 4),
                                                                   np.round(total_metrics_dict[metric][2], 4)))
    print("###################### TEST REPORT ######################\n")

    if args.train_data_type == args.test_data_type:
        test_results_save_path = os.path.join(model_dirs, 'test_reports', 'test_report(EPOCH {})_TotalResults.txt'.format(args.final_epoch))
    else:
        test_results_save_path = os.path.join(model_dirs, 'test_reports', 'Generalizability test_reports({}->{})_TotalResults.txt'.format(args.train_data_type, args.test_data_type))

    f = open(test_results_save_path, 'w')

    f.write("###################### TEST REPORT ######################\n")
    for metric in total_metrics_dict.keys():
        f.write("Trial Mean | Medium {}   :\t {} | {} ({}) [{} | {} | {}]\n".format(metric,
                                                                   np.round(np.mean(total_metrics_dict[metric]), 4),
                                                                   np.round(np.median(total_metrics_dict[metric]), 4),
                                                                   np.round(np.std(total_metrics_dict[metric]), 4),
                                                                   np.round(total_metrics_dict[metric][0], 4),
                                                                   np.round(total_metrics_dict[metric][1], 4),
                                                                   np.round(total_metrics_dict[metric][2], 4)))
    f.write("###################### TEST REPORT ######################\n")

    f.close()

    print("test results txt file is saved at {}".format(test_results_save_path))