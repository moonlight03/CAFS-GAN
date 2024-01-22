"""
TUNIT: Truly Unsupervised Image-to-Image Translation
Copyright (c) 2020-present NAVER Corp.
MIT license 
"""
from torch import nn
import torch.nn.functional as F
import torch

try:
    from models.blocks_stylegan2 import Conv2dBlock, FRN, SelfAttention
except:
    from blocks_stylegan2 import Conv2dBlock, FRN, SelfAttention


cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'vgg19cut': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'N'],
}


class GuidingNet(nn.Module):
    def __init__(self, img_size=64, output_k={'cont': 128, 'disc': 10}):
        super(GuidingNet, self).__init__()
        # network layers setting
        self.features = make_layers(cfg['vgg11'], True)
        self.SA = SelfAttention(512)
        self.disc = nn.Linear(512, output_k['disc'])
        self.cont = nn.Linear(512, output_k['cont'])

        self._initialize_weights()

    def forward(self, x, sty=False):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        flat = x.view(x.size(0), -1)
        cont = self.cont(flat)
        if sty:
            return cont
        disc = self.disc(flat)
        return {'cont': cont, 'disc': disc}

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def moco(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        flat = x.view(x.size(0), -1)
        cont = self.cont(flat)
        return cont

    def moco1(self, x):
        x = self.features(x)
        return x
    def moco_attention(self, x, x_aug):

        x = self.features(x)
        x_aug = self.features(x_aug)
        out = self.SA(x, x_aug)

        out = F.adaptive_avg_pool2d(out, (1, 1))
        flat = out.view(out.size(0), -1)
        out = self.cont(flat)
        return out

    def iic(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        flat = x.view(x.size(0), -1)
        disc = self.disc(flat)
        return disc


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
            else:
                layers += [conv2d, nn.ReLU(inplace=False)]  # 原始版只有两项，没有norm
            in_channels = v
    return nn.Sequential(*layers)


if __name__ == '__main__':
    import torch
    C = GuidingNet(64)
    x_in = torch.randn(4, 3, 64, 64)
    sty = C.moco(x_in)
    cls = C.iic(x_in)
    print(sty.shape, cls.shape)
