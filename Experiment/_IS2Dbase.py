import os
import sys
import random
from datetime import datetime

import torch
from torch.utils.data import DataLoader

import numpy as np

from IS2D_models import IS2D_model, model_to_device

from utils.misc import CSVLogger, byte_transform
from utils.get_functions import get_deivce, get_optimizer, get_scheduler, get_save_path

from dataset.customdataset import CustomDataset

class BaseSegmentationExperiment(object):
    def __init__(self, args):
        super(BaseSegmentationExperiment, self).__init__()

        self.args = args
        self.args.device = get_deivce()
        if args.fix_seed: self.fix_seed()
        self.history_generator()
        self.start, self.end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.inference_time_list = []
        self.scaler = torch.cuda.amp.GradScaler()

        print("STEP1-1. Load Train {} Dataset Loader...".format(args.train_data_type))
        self.train_loader = self.dataloader_generator(self.args.train_dataset_dir, self.args.train_batch_size, mode='train')

        print("STEP1-2. Load Test {} Dataset Loader...".format(args.test_data_type))
        self.test_loader = self.dataloader_generator(self.args.test_dataset_dir, self.args.test_batch_size, mode='test')

        print("STEP2. Load 2D Image Segmentation Model {}...".format(args.model_name))
        self.model = IS2D_model(args)
        self.model = model_to_device(self.args, self.model)

        print("STEP3. Load Optimizer {}...".format(args.optimizer_name))
        self.optimizer = get_optimizer(args, self.model)

        print("STEP4. Load LRS {}...".format(args.LRS_name))
        self.scheduler = get_scheduler(args, self.optimizer, len(self.train_loader))

        if self.args.efficiency_analysis:
            from thop import profile
            inp = torch.randn((1, self.args.num_channels, self.args.image_size, self.args.image_size))
            tap = torch.randn((1, self.args.num_classes, self.args.image_size, self.args.image_size))
            macs, params = profile(self.model.cuda(), inputs=(inp.cuda(), tap.cuda()))
            print("FLOPs : {}GB".format(byte_transform(macs * 2, to='g')))
            print("Params : {}MB".format(byte_transform(params, to='m')))
            sys.exit()

        if self.args.train:
            print("STEP5. Make Train Log File...")
            now = datetime.now()
            model_dirs = get_save_path(args)
            if not os.path.exists(os.path.join(model_dirs, 'logs')): os.makedirs(os.path.join(model_dirs, 'logs'))
            filename = os.path.join(model_dirs, 'logs', '{}-{}-{} {}:{}:{} log.csv'.format(now.year, now.month, now.day, now.hour, now.minute, now.second))
            self.csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_loss', 'test_loss'], filename=filename)

    def print_params(self):
        print("\ntrain data type (#Samples) : {} ({})".format(self.args.train_data_type, self.train_loader.dataset.__len__()))
        print("test data type (#Samples) : {} ({})".format(self.args.test_data_type, self.test_loader.dataset.__len__()))
        print("Trial Number : {}".format(self.args.current_trial))
        print("model : {}".format(self.args.model_name))
        print("optimizer : {}".format(self.optimizer))
        print("learning rate : {}".format(self.args.lr))
        print("learning rate scheduler : {}".format(self.args.LRS_name))
        print("final epoch : {}".format(self.args.final_epoch))
        print("train batch size : {}".format(self.args.train_batch_size))
        print("data augmentation: {}".format(self.transform_generator('train')))
        print("image size : ({}, {}, {})".format(self.args.image_size, self.args.image_size, self.args.num_channels))
        print("pytorch_total_params : {}".format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

    def fix_seed(self):
        random.seed(4321)
        np.random.seed(4321)
        torch.manual_seed(4321)
        torch.cuda.manual_seed(4321)
        torch.cuda.manual_seed_all(4321)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        print("your seed is fixed to '4321' with reproducibility")

    def history_generator(self):
        self.history = dict()
        self.history['train_loss'] = list()
        self.history['val_loss'] = list()

    def current_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def forward(self, data, mode):
        if self.args.her_image:
            image, nuclear_mask, her_image = data
            data = (image.to(self.args.device), nuclear_mask.to(self.args.device), her_image.to(self.args.device))
        else:
            image, nuclear_mask = data
            data = (image.to(self.args.device), nuclear_mask.to(self.args.device))

        with torch.cuda.amp.autocast():
            output_dict = self.model(data, mode)

        return output_dict

    def backward(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.args.LRS_name == 'CALRS': self.scheduler.step()

    def dataloader_generator(self, dataset_dir, batch_size, mode):
        image_transform, target_transform = self.transform_generator(mode)
        dataset = CustomDataset(dataset_dir=dataset_dir, image_transform=image_transform, target_transform=target_transform, mode=mode, her_image=True)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=self.args.num_workers, pin_memory=True)

        return data_loader