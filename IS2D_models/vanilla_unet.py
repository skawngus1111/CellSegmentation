import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module) :
    def __init__(self, in_dim, n_class, num_filters):
        super(UNet, self).__init__()

        self.criterion = nn.BCEWithLogitsLoss()

        self.in_dim = in_dim
        self.n_class = n_class
        self.num_filters = num_filters

        act_fn = nn.LeakyReLU(0.2, inplace=True)

        # Encoding Parts
        self.down1 = conv_block_2(self.in_dim, self.num_filters, act_fn)
        self.pool1 = maxpool()
        self.down2 = conv_block_2(self.num_filters * 1, self.num_filters * 2, act_fn)
        self.pool2 = maxpool()
        self.down3 = conv_block_2(self.num_filters * 2, self.num_filters * 4, act_fn)
        self.pool3 = maxpool()
        self.down4 = conv_block_2(self.num_filters * 4, self.num_filters * 8, act_fn)
        self.pool4 = maxpool()

        self.bridge = conv_block_2(self.num_filters * 8, self.num_filters * 16, act_fn)

        # Decoding Parts
        self.trans1 = conv_trans_block(self.num_filters * 16, self.num_filters * 8, act_fn)
        self.up1    = conv_block_2(self.num_filters * 16, self.num_filters * 8, act_fn)
        self.trans2 = conv_trans_block(self.num_filters * 8, self.num_filters * 4, act_fn)
        self.up2    = conv_block_2(self.num_filters * 8, self.num_filters * 4, act_fn)
        self.trans3 = conv_trans_block(self.num_filters * 4, self.num_filters * 2, act_fn)
        self.up3    = conv_block_2(self.num_filters * 4, self.num_filters * 2, act_fn)
        self.trans4 = conv_trans_block(self.num_filters * 2, self.num_filters * 1, act_fn)
        self.up4    = conv_block_2(self.num_filters * 2, self.num_filters * 1, act_fn)

        # output block
        self.out = nn.Sequential(
            nn.Conv2d(self.num_filters, self.n_class, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

    def forward(self, image, target, mode='train'):
        if image.size()[1] == 1: image = image.repeat(1, 3, 1, 1)

        # feature encoding
        down1 = self.down1(image)
        pool1 = self.pool1(down1)
        down2 = self.down2(pool1)
        pool2 = self.pool2(down2)
        down3 = self.down3(pool2)
        pool3 = self.pool3(down3)
        down4 = self.down4(pool3)
        pool4 = self.pool4(down4)

        bridge = self.bridge(pool4)

        # feature decoding
        trans1  = self.trans1(bridge)
        concat1 = torch.cat([trans1, down4], dim=1)
        up1     = self.up1(concat1)

        trans2  = self.trans2(up1)
        concat2 = torch.cat([trans2, down3], dim=1)
        up2     = self.up2(concat2)

        trans3  = self.trans3(up2)
        concat3 = torch.cat([trans3, down2], dim=1)
        up3     = self.up3(concat3)

        trans4  = self.trans4(up3)
        concat4 = torch.cat([trans4, down1], dim=1)
        up4     = self.up4(concat4)

        out = self.out(up4)
        output_dict = self._calculate_criterion(out, target)

        return output_dict

    def _calculate_criterion(self, y_pred, y_true):
        loss = F.binary_cross_entropy_with_logits(y_pred, y_true)

        output_dict = {'loss': loss, 'output': y_pred, 'target': y_true}

        return output_dict


def conv_block_2(in_dim, out_dim, act_fn) :
    model = nn.Sequential(
        conv_block(in_dim, out_dim, act_fn),
        nn.Conv2d(out_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(out_dim)
    )

    return model

def conv_trans_block(in_dim, out_dim, act_fn) :
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
        nn.BatchNorm2d(out_dim), act_fn
    )

    return model

def maxpool() :
    pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))

    return pool

def conv_block(in_dim, out_dim, act_fn) :
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(out_dim),
        act_fn
    )

    return model