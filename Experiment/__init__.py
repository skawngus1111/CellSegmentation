import os
import sys

from .cellsegmentationexperiment import CellSegmentationExperiment

def dataset_argument(args):
    try:
        args.train_dataset_dir = os.path.join(args.data_path, args.train_data_type)
        args.test_dataset_dir = os.path.join(args.data_path, args.test_data_type)
    except TypeError:
        print("join() argument must be str, bytes, or os.PathLike object, not 'NoneType'")
        sys.exit()

    print("Train dataset directory : ", args.train_dataset_dir)
    print("Test dataset directory : ", args.test_dataset_dir)

    # Dataset Argument
    args.num_channels = 3
    args.image_size = 256
    args.num_classes = 1
    args.mean = [0.485, 0.456, 0.406]
    args.std = [0.229, 0.224, 0.225]
    # args.metric_list = ['AJI', 'DSC', 'IoU', 'Precision', 'Recall', 'F1-Score']
    args.metric_list = ['IoU', 'DSC', 'Recall', 'Precision', 'Specificity']

    # Training Argument
    args.train_batch_size = 4
    args.test_batch_size = 4
    args.start_epoch = 1
    args.final_epoch = 200

    # Optimizer Argument
    args.optimizer_name = 'AdamW'
    args.lr = 5e-4
    args.weight_decay = 5e-4
    args.LRS_name = 'CALRS'

    return args