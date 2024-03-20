import os

import torch

from IS2D_models import model_to_device
from utils.get_functions import get_save_path

def load_model(args, model) :
    model_dirs = get_save_path(args)

    load_path = os.path.join(model_dirs, 'model_weights/model_weight(EPOCH {})_Trial{}.pth.tar'.format(args.final_epoch, args.current_trial))

    print("Your model is loaded from {}.".format(load_path))
    checkpoint = torch.load(load_path)
    print(".pth.tar keys() =  {}.".format(checkpoint.keys()))

    model.load_state_dict(checkpoint['model_state_dict'])
    args.start_epoch = checkpoint['current_epoch']

    model = model_to_device(args, model)

    return model