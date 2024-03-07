from time import time

import torch
import torchvision.transforms as transforms

import numpy as np
from tqdm import tqdm

from ._IS2Dbase import BaseSegmentationExperiment
from utils.calculate_metrics import CellSegmentation_Metrics_Calculator
from utils.load_functions import load_model

class CellSegmentationExperiment(BaseSegmentationExperiment):
    def __init__(self, args):
        super(CellSegmentationExperiment, self).__init__(args)

        self.metrics_calculator = CellSegmentation_Metrics_Calculator(args.metric_list)
        self.count = 1

    def fit(self):
        self.print_params()
        if self.args.train:
            for epoch in tqdm(range(self.args.start_epoch, self.args.final_epoch + 1)):
                print('\n============ EPOCH {}/{} ============\n'.format(epoch, self.args.final_epoch))
                if epoch % 10 == 0: self.print_params()
                epoch_start_time = time()

                print("TRAINING")
                train_results = self.train_epoch(epoch)

                print("EVALUATE")
                val_results = self.val_epoch(epoch)

                total_epoch_time = time() - epoch_start_time
                m, s = divmod(total_epoch_time, 60)
                h, m = divmod(m, 60)

                self.history['train_loss'].append(train_results)
                self.history['val_loss'].append(val_results)

                row = {'epoch': str(epoch),
                       'train_loss': str(train_results),
                       'test_loss': str(val_results)}
                self.csv_logger.writerow(row)

                print('\nEpoch {}/{} : train loss {:.6f} | val loss {:.6f} | current lr {:.8f} | took {} h {} m {} s'.format(
                    epoch, self.args.final_epoch, np.round(train_results, 4), np.round(val_results, 4),
                    self.current_lr(self.optimizer), int(h), int(m), int(s)))

            print("INFERENCE")
            test_results = self.inference()

            return self.model, self.optimizer, self.history, test_results

        else :
            print("INFERENCE")
            self.model = load_model(self.args, self.model)
            test_results = self.inference()

            return test_results

    def train_epoch(self, epoch):
        self.model.train()

        total_loss, total = 0., 0

        for batch_idx, data in enumerate(self.train_loader):
            output_dict = self.forward(data, mode='train')
            self.backward(output_dict['loss'])

            total_loss += output_dict['loss'].item() * output_dict['target'].size(0)
            total += output_dict['target'].size(0)

            if (batch_idx + 1) % self.args.step == 0 or (batch_idx + 1) == len(self.train_loader):
                print("Epoch {} | batch_idx : {}/{}({}%) COMPLETE | loss : {}".format(
                    epoch, batch_idx + 1, len(self.train_loader), np.round((batch_idx + 1) / len(self.train_loader) * 100.0, 2),
                    total_loss / total))

        train_loss = total_loss / total

        return train_loss

    def val_epoch(self, epoch):
        self.model.eval()

        total_loss, total = 0., 0

        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_loader):
                if (batch_idx + 1) % self.args.step == 0:
                    print("EPOCH {} | {}/{}({}%) COMPLETE".format(epoch, batch_idx + 1, len(self.test_loader), np.round((batch_idx + 1) / len(self.test_loader) * 100), 4))

                output_dict = self.forward(data, mode='test')

                total_loss += output_dict['loss'].item() * output_dict['target'].size(0)
                total += output_dict['target'].size(0)

        test_loss = total_loss / total

        return test_loss

    def inference(self):
        self.model.eval()

        total_metrics_dict = self.metrics_calculator.total_metrics_dict

        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_loader):
                self.start.record()
                output_dict = self.forward(data, mode='test')
                self.end.record()
                torch.cuda.synchronize()
                self.inference_time_list.append(self.start.elapsed_time(self.end))

                for idx, (target_, output_) in enumerate(zip(output_dict['target'], output_dict['output'])):
                    predict = torch.sigmoid(output_).squeeze()
                    if self.args.plot_inference:
                        self.plot_image_and_prediction(data['image'][idx], target_, predict, self.count)
                        self.count += 1

                    metrics_dict = self.metrics_calculator.get_metrics_dict(predict, target_)

                    for metric in self.metrics_calculator.metric_list:
                        total_metrics_dict[metric].append(metrics_dict[metric])

        for metric in self.metrics_calculator.metric_list:
            total_metrics_dict[metric] = np.round(np.mean(total_metrics_dict[metric]), 4)

        print(total_metrics_dict)

        print("Mean Inference Time (ms) : {} ({})".format(np.round(np.mean(self.inference_time_list[1:]), 2),
                                                          np.round(np.std(self.inference_time_list[1:]), 2)))

        return total_metrics_dict

    def transform_generator(self, mode):
        if mode == 'train':
            transform_list = [
                transforms.ToPILImage(),
                transforms.Resize((self.args.image_size, self.args.image_size)),
                transforms.RandomRotation((-5, 5), expand=False),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.args.mean, std=self.args.std)
            ]

            target_transform_list = [
                transforms.ToPILImage(),
                transforms.Resize((self.args.image_size, self.args.image_size)),
                transforms.RandomRotation((-5, 5), expand=False),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        else:
            transform_list = [
                transforms.ToPILImage(),
                transforms.Resize((self.args.image_size, self.args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.args.mean, std=self.args.std)
            ]

            target_transform_list = [
                transforms.ToPILImage(),
                transforms.Resize((self.args.image_size, self.args.image_size)),
                transforms.ToTensor(),
            ]

        return transforms.Compose(transform_list), transforms.Compose(target_transform_list)