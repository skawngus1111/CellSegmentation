import argparse
import warnings
warnings.filterwarnings('ignore')

from utils.get_functions import get_save_path
from utils.save_functions import save_result, save_metrics, save_total_trial_result
from Experiment import dataset_argument, CellSegmentationExperiment

def main(args):
    print("Hello! We start experiment for COMPUTER VISION for MICROSCOPY IMAGE ANALYSIS (CVMI)!")

    args = dataset_argument(args)

    print("Training Arguments : {}".format(args))

    experiment = CellSegmentationExperiment(args)

    if args.train:
        model, optimizer, history, test_results = experiment.fit()
        save_result(args, model, optimizer, history, test_results)
    else:
        test_results = experiment.fit()
        model_dirs = get_save_path(args)

        print("Save {} Model Test Results...".format(args.model_name))
        save_metrics(args, test_results, model_dirs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Following are the arguments that can be passed form the terminal itself!')
    parser.add_argument('--data_path', type=str, default='/media/jhnam0514/68334fe0-2b83-45d6-98e3-76904bf08127/home/namjuhyeon/Desktop/LAB/AwesomeDeepLearning/dataset/IS2D_dataset/BioMedicalDataset')
    parser.add_argument('--save_path', type=str, default='/media/jhnam0514/68334fe0-2b83-45d6-98e3-76904bf08127/home/namjuhyeon/Desktop/LAB/CVMI2024')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--train_data_type', type=str, required=False, choices=['CoNSeP', 'PanNuke'])
    parser.add_argument('--test_data_type', type=str, required=False, choices=['CoNSeP', 'PanNuke'])
    parser.add_argument('--plot_inference', default=False, action='store_true')
    parser.add_argument('--efficiency_analysis', default=False, action='store_true')
    parser.add_argument('--fix_seed', default=False, action='store_true')

    # Train parameter
    parser.add_argument('--model_name', type=str, required=False, choices=['VanillaUNet', 'Ours'])
    parser.add_argument('--train', default=False, action='store_true')

    # Ours parameter
    parser.add_argument('--our_method_configuration', type=str, required=False, default='test')
    parser.add_argument('--model_scale', type=str, required=False, default='Large')
    parser.add_argument('--cnn_backbone', type=str, default=None, choices=['resnet50', 'res2net50_v1b_26w_4s', 'resnest50'])
    parser.add_argument('--transformer_backbone', type=str, default=None, choices=['pvt_v2_b2'])
    parser.add_argument('--her_image', default=False, action='store_true')

    # Print parameter
    parser.add_argument('--step', type=int, default=10)

    args = parser.parse_args()

    # for model_scale in ['Large', 'Base', 'Tiny']:
    #     args.model_scale = model_scale
    for train_data_type in ['CoNSeP', 'PanNuke']:
        args.train_data_type = train_data_type

        for current_trial in range(1, 4):
            args.current_trial = current_trial
            args.train = True
            args.test_data_type = args.train_data_type
            main(args)

            for test_data_type in ['CoNSeP', 'PanNuke']:
                args.train = False
                args.test_data_type = test_data_type
                main(args)

        # Save total trial result
        args = dataset_argument(args)
        for test_data_type in ['CoNSeP', 'PanNuke']:
            args.train = False
            args.test_data_type = test_data_type
            for current_trial in range(1, 4):
                args.current_trial = current_trial

            save_total_trial_result(args)