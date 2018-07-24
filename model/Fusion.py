import torch.nn as nn
from torchvision import models


class VGG(nn.Module):
    def __init__(self, input_channels):
        feats = list(models.vgg16(pretrained=True).features.children())
        feats[0] = nn.Conv2d(input_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv1 = nn.Sequential(*feats[0:10])
        self.conv2 = nn.Sequential(*feats[10:17])
        self.conv3 = nn.Sequential(*feats[17:24])
        self.conv4 = nn.Sequential(*feats[24:31])

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        feats3 = self.conv3(x)
        feats4 = self.conv4(feats3)
        feats5 = self.conv5(feats4)
        return feats3, feats4, feats5


class Extract(nn.Module):
    def __init__(self, input_channels):
        super(Extract, self).__init__()
        self.tem = VGG(input_channels)

    def forward(self, x):
        x = self.tem(x)
        return x
