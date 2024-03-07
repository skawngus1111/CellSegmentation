import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from IS2D_models import load_cnn_backbone_model, load_transformer_backbone_model

class DepthwiseSepConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int=1,
                 padding: int=0,
                 dilation: int=1,) -> None:
        super(DepthwiseSepConv2d, self).__init__()

        self.depthwise_conv = nn.Conv2d(in_channels=in_channels,
                                        out_channels=in_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        dilation=dilation,
                                        groups=in_channels)
        self.bn = nn.BatchNorm2d(in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        x = self.bn(x)
        x = self.pointwise_conv(x)

        return x

class HER_Guided_Self_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding) -> None:
        super(HER_Guided_Self_Attention, self).__init__()

        self.to_q = DepthwiseSepConv2d(in_channels, out_channels, kernel_size, stride, padding) # Query from input feature
        self.to_k = DepthwiseSepConv2d(in_channels, out_channels, kernel_size, stride, padding) # Key from her feature
        self.to_v = DepthwiseSepConv2d(in_channels, out_channels, kernel_size, stride, padding) # Value from input feature

    def forward(self, x, her_feature):
        q = self.to_q(x)
        k = self.to_k(her_feature)
        v = self.to_v(x)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * (1 / q.shape[-1]) ** 0.5
        attn = dots.softmax(dim=-1)
        import matplotlib.pyplot as plt
        plt.imshow(attn[0, 0].detach().cpu().numpy())
        plt.show()

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)

        return out

class HER_Guided_Channel_Attention(nn.Module):
    def __init__(self,
                 in_channels: int,
                 reduction: int=16) -> None:
        super(HER_Guided_Channel_Attention, self).__init__()

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0, bias=False))

        self.average_channel_pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, her_feature):
        her_feature_pool = self.average_channel_pooling(her_feature)
        her_feature_channel_weight = self.fc(her_feature_pool)
        her_feature_channel_weight = torch.sigmoid(her_feature_channel_weight)

        out = x * her_feature_channel_weight

        return out

class UpsampleBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 skip_connection_channels,
                 reduction: int=16) -> None:
        super(UpsampleBlock, self).__init__()

        in_channels = in_channels + skip_connection_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

        # self.her_guided_self_attention = HER_Guided_Self_Attention(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.her_guided_channel_attention = HER_Guided_Channel_Attention(out_channels, reduction=reduction)

    def forward(self, x, skip_connection=None, her_feature=None):
        x = F.interpolate(x, size=None, scale_factor=2, mode='bilinear', align_corners=None)

        # Concatenate with features from encoder stage
        x = torch.cat([x, skip_connection], dim=1)
        x = self.conv(x)

        # HER feature-based Guided Self-Attention
        # her_enhanced_feature = self.her_guided_self_attention(x, her_feature)
        her_enhanced_feature = self.her_guided_channel_attention(x, her_feature)

        return her_enhanced_feature

class SubDecoder(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 scale_factor,
                 interpolation_mode='bilinear'):
        super(SubDecoder, self).__init__()

        self.output_conv = nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=interpolation_mode)

    def forward(self, x):
        x = self.output_conv(x)
        x = self.upsample(x)

        return x

class Ours(nn.Module):
    def __init__(self,
                 num_channels: int=3,
                 num_classes: int=1,
                 model_scale: str='Large',
                 cnn_backbone: str='resnet50',
                 transformer_backbone: str='pvt_v2_b2') -> None:
        super(Ours, self).__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes
        self.cnn_backbone = cnn_backbone
        self.transformer_backbone = transformer_backbone

        # STEP1. Load ImageNet pre-trained backbone
        if cnn_backbone in ['resnet50', 'res2net50_v1b_26w_4s', 'resnest50']:
            self.backbone = load_cnn_backbone_model(backbone_name=cnn_backbone, pretrained=True)
            self.in_channels = 2048
            self.skip_channel_list = [1024, 512, 256, 64]
            self.decoder_channel_list = [256, 128, 64, 32]

        if transformer_backbone in ['pvt_v2_b2']:
            self.backbone = load_transformer_backbone_model(backbone_name=transformer_backbone, pretrained=True)
            self.in_channels = 2048
            self.skip_channel_list = [512, 320, 128, 64]
            self.decoder_channel_list = [256, 128, 64, 32]

            self.feature_embedding = nn.Sequential(
                nn.Conv2d(512, self.in_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.in_channels), nn.ReLU(inplace=True))

        if model_scale == 'Large': self.skip_channel_down_list = [1024, 512, 256, 64]
        elif model_scale == 'Base': self.skip_channel_down_list = [64, 64, 64, 64]
        elif model_scale == 'Tiny': self.skip_channel_down_list = [1, 1, 1, 1]
        else:
            print("Wrong model scale settings...")
            sys.exit()

        # Define her image encoder for fusing in decoder
        self.her_image_encoder1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.her_image_encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.her_image_encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.her_image_encoder4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # STEP2. Skip Connection
        self.skip_connection1 = nn.Sequential(
            nn.Conv2d(self.skip_channel_list[0], self.skip_channel_down_list[0], kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.skip_channel_down_list[0]), nn.ReLU(inplace=True))
        self.skip_connection2 = nn.Sequential(
            nn.Conv2d(self.skip_channel_list[1], self.skip_channel_down_list[1], kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.skip_channel_down_list[1]), nn.ReLU(inplace=True))
        self.skip_connection3 = nn.Sequential(
            nn.Conv2d(self.skip_channel_list[2], self.skip_channel_down_list[2], kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.skip_channel_down_list[2]), nn.ReLU(inplace=True))
        self.skip_connection4 = nn.Sequential(
            nn.Conv2d(self.skip_channel_list[3], self.skip_channel_down_list[3], kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.skip_channel_down_list[3]), nn.ReLU(inplace=True))

        # STEP3. Decoding
        self.decoder_stage1 = UpsampleBlock(self.in_channels, self.decoder_channel_list[0], self.skip_channel_down_list[0])
        self.decoder_stage2 = UpsampleBlock(self.decoder_channel_list[0], self.decoder_channel_list[1], self.skip_channel_down_list[1])
        self.decoder_stage3 = UpsampleBlock(self.decoder_channel_list[1], self.decoder_channel_list[2], self.skip_channel_down_list[2])
        self.decoder_stage4 = UpsampleBlock(self.decoder_channel_list[2], self.decoder_channel_list[3], self.skip_channel_down_list[3])

        # STEP4. Output
        self.sub_Decoder_stage1 = SubDecoder(self.decoder_channel_list[0], self.num_classes, scale_factor=16, interpolation_mode='bilinear')
        self.sub_Decoder_stage2 = SubDecoder(self.decoder_channel_list[1], self.num_classes, scale_factor=8, interpolation_mode='bilinear')
        self.sub_Decoder_stage3 = SubDecoder(self.decoder_channel_list[2], self.num_classes, scale_factor=4, interpolation_mode='bilinear')
        self.sub_Decoder_stage4 = SubDecoder(self.decoder_channel_list[3], self.num_classes, scale_factor=2, interpolation_mode='bilinear')

    def forward(self, data, target, mode='train'):
        # x, target, her_image = data
        x = data
        her_image = data
        if x.size()[1] == 1: x = x.repeat(1, 3, 1, 1)

        # feature encoding
        features, x = self.backbone.forward_feature(x, out_block_stage=4)

        # HER encoding
        her_feature_1 = self.her_image_encoder1(her_image)
        her_feature_2 = self.her_image_encoder2(her_feature_1)
        her_feature_3 = self.her_image_encoder3(her_feature_2)
        her_feature_4 = self.her_image_encoder4(her_feature_3)

        # feature decoding
        x1 = self.decoder_stage1(x, self.skip_connection1(features[0]), her_feature_4)
        x2 = self.decoder_stage2(x1, self.skip_connection2(features[1]), her_feature_3)
        x3 = self.decoder_stage3(x2, self.skip_connection3(features[2]), her_feature_2)
        x4 = self.decoder_stage4(x3, self.skip_connection4(features[3]), her_feature_1)

        # output block
        if mode == 'train':
            stage1_output = self.sub_Decoder_stage1(x1)
            stage2_output = self.sub_Decoder_stage2(x2)
            stage3_output = self.sub_Decoder_stage3(x3)
            stage4_output = self.sub_Decoder_stage4(x4)

            output = [stage1_output, stage2_output, stage3_output, stage4_output]

            output_dict = self._calculate_criterion(output, target, mode)
        else:
            output = self.sub_Decoder_stage4(x4)
            output_dict = self._calculate_criterion(output, target, mode)

        return output_dict

    def _calculate_criterion(self, y_pred, y_true, mode='train'):
        # loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
        # loss = self.structure_loss(y_pred, y_true)

        if mode == 'train':
            loss = 0.0
            for y_pred_ in y_pred:
                loss += F.binary_cross_entropy_with_logits(y_pred_, y_true)
        else:
            loss = F.binary_cross_entropy_with_logits(y_pred, y_true)

        output_dict = {'loss': loss, 'output': y_pred, 'target': y_true}

        return output_dict

    def structure_loss(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        return (wbce + wiou).mean()