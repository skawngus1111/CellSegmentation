import sys

import torch
import torch.utils.model_zoo as model_zoo

import IS2D_models.backbone as backbone

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'res2net50_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth',
    'resnest50': 'https://github.com/zhanghang1989/ResNeSt/releases/download/weights_step1/528c19ca-resnest50.pth'
}

def IS2D_model(args):
    if args.model_name == 'VanillaUNet' :
        from IS2D_models.vanilla_unet import UNet
        model = UNet(args.num_channels, args.num_classes, num_filters=32)
    elif args.model_name == 'Ours':
        from IS2D_models.ours import Ours
        model = Ours(num_channels=args.num_channels,
                     num_classes=args.num_classes,
                     model_scale=args.model_scale,
                     cnn_backbone=args.cnn_backbone,
                     transformer_backbone=args.transformer_backbone)
    return model

def model_to_device(args, model):
    model = model.to(args.device)

    return model

def load_transformer_backbone_model(backbone_name, pretrained=False):
    if backbone_name == 'pvt_v2_b2':
        from IS2D_models.backbone.transformer.pvt_v2 import pvt_v2_b2
        model = pvt_v2_b2()
    else:
        print("Invalid Transformer backbone")
        sys.exit()

    if pretrained:
        load_pretrained_path = 'IS2D_models/backbone/weights/{}.pth'.format(backbone_name)
        model.load_state_dict(torch.load(load_pretrained_path))

        print("Complete loading your pretrained Transformer backbone {}".format(backbone_name))

    model.fc = torch.nn.Identity()

    return model

def load_cnn_backbone_model(backbone_name, pretrained=False):
    if backbone_name == 'resnet50':
        from IS2D_models.backbone.resnet import ResNet
        model = ResNet(backbone.resnet.Bottleneck, [3, 4, 6, 3])
    elif backbone_name=='res2net50_v1b_26w_4s':
        from IS2D_models.backbone.res2net import Res2Net
        model = Res2Net(backbone.res2net.Bottle2Neck, [3, 4, 6, 3], baseWidth=26, scale=4)
    elif backbone_name=='resnest50':
        from IS2D_models.backbone.resnest import ResNeSt
        model = ResNeSt(backbone.resnest.Bottleneck, [3, 4, 6, 3],
                        radix=2, groups=1, bottleneck_width=64,
                        deep_stem=True, stem_width=32, avg_down=True,
                        avd=True, avd_first=False)
    else:
        print("Invalid CNN backbone")
        sys.exit()


    if pretrained:
        if backbone_name == 'resnest50':
            _url_format = 'https://github.com/zhanghang1989/ResNeSt/releases/download/weights_step1/{}-{}.pth'

            _model_sha256 = {name: checksum for checksum, name in [
                ('528c19ca', 'resnest50'),
                ('22405ba7', 'resnest101'),
                ('75117900', 'resnest200'),
                ('0cc87c48', 'resnest269'),
            ]}

            def short_hash(name):
                if name not in _model_sha256:
                    raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
                return _model_sha256[name][:8]

            resnest_model_urls = {name: _url_format.format(name, short_hash(name)) for
                                  name in _model_sha256.keys()
                                  }

            model.load_state_dict(torch.hub.load_state_dict_from_url(
                resnest_model_urls['resnest50'], progress=True, check_hash=True))
        else:
            model.load_state_dict(model_zoo.load_url(model_urls[backbone_name]))

        print("Complete loading your pretrained CNN backbone {}".format(backbone_name))

    model.fc = torch.nn.Identity()

    return model