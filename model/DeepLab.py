import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision import models
affine_par = True


class Conv_BN_ReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 affine=True, inplace=False):
        super(Conv_BN_ReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(num_features=out_channels, affine=affine)
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inChannels, midChannels, stride=1, dilation_=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, midChannels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(midChannels, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        self.conv2 = nn.Conv2d(midChannels, midChannels, kernel_size=3, stride=1,
                               padding=dilation_, bias=False, dilation=dilation_)
        self.bn2 = nn.BatchNorm2d(midChannels, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False

        self.conv3 = nn.Conv2d(midChannels, midChannels*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(midChannels*4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False

        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, InputChannels, dilation=[1, 1, 1], layers=[3, 4, 6, 3]):
        super(ResNet, self).__init__()
        self.inplanes = 1024
        feats = list(models.resnet50(pretrained=True).children())
        feats[0] = nn.Conv2d(InputChannels, 64, kernel_size=7, stride=2, padding=3)
        self.layer1 = nn.Sequential(*feats[0:3])
        self.layer2 = nn.Sequential(*feats[3:5])
        self.layer3 = feats[5]
        self.layer4 = feats[6]

        # self.conv1 = nn.Conv2d(InputChannels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        # # for i in self.bn1.parameters():
        # #     i.requires_grad = False
        # self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        # self.layer1 = self.make_layer(64, layers[0])
        # self.layer2 = self.make_layer(128, layers[1], stride=2)
        # self.layer3 = self.make_layer(256, layers[2], stride=2)
        self.layer5 = self.make_layer(512, layers[3], stride=1, dilation=dilation)

    def make_layer(self, planes, layer, stride=1, dilation=[1], downsample=True):
        layers = []
        if downsample:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*Bottleneck.expansion, affine=affine_par)
            )
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False

        layers.append(Bottleneck(self.inplanes, planes, stride=stride, dilation_=dilation[0], downsample=downsample))
        self.inplanes = planes * Bottleneck.expansion
        for i in range(1, layer):
            if len(dilation) <= i:
                dilation.append(1)
            layers.append(Bottleneck(self.inplanes, planes, dilation_=dilation[i]))
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.layer1(x)
        x_low = self.layer2(x)
        x = self.layer3(x_low)
        x = self.layer4(x)
        x = self.layer5(x)

        return x, x_low


class ASPP(nn.Module):
    def __init__(self, channels):
        super(ASPP, self).__init__()
        self.conv_1x1 = nn.Conv2d(channels, 256, kernel_size=1, stride=1)
        self.bn_1x1 = nn.BatchNorm2d(256)
        self.relu_1x1 = nn.ReLU(inplace=True)
        self.conv_3x3_6 = Conv_BN_ReLU(channels, 256, kernel_size=3, stride=1, dilation=6, padding=6)
        self.conv_3x3_12 = Conv_BN_ReLU(channels, 256, kernel_size=3, stride=1, dilation=12, padding=12)
        self.conv_3x3_18 = Conv_BN_ReLU(channels, 256, kernel_size=3, stride=1, dilation=18, padding=18)
        self.image_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv_1x1_reduce = Conv_BN_ReLU(channels, 256, kernel_size=1, stride=1)
        self.conv_1x1_cat = Conv_BN_ReLU(256*5, 256, kernel_size=1, stride=1)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        out_pool = self.conv_1x1_reduce(self.image_pool(x))
        out_pool = f.upsample(input=out_pool, size=(h, w), mode='bilinear')
        out_1 = self.conv_1x1(x)
        out_6 = self.conv_3x3_6(x)
        out_12 = self.conv_3x3_12(x)
        out_18 = self.conv_3x3_18(x)
        out = torch.cat((out_pool, out_1, out_6, out_12, out_18), dim=1)
        out = self.conv_1x1_cat(out)

        return out


class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv = nn.Conv2d(2048, 8, kernel_size=1)
        self.fc = nn.Linear(32, 4)
        self.activate = nn.Softmax(dim=1)

    def forward(self, x):
        tem_features = self.conv(self.pool(x[0]))
        hum_features = self.conv(self.pool(x[1]))
        x_wind_features = self.conv(self.pool(x[2]))
        y_wind_features = self.conv(self.pool(x[3]))
        features = torch.cat((tem_features, hum_features, x_wind_features, y_wind_features), dim=1)
        features = features.view(-1, 32)
        out = self.activate(self.fc(features))

        return out


class Classify(nn.Module):
    def __init__(self, num_class):
        super(Classify, self).__init__()
        self.conv = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(256, affine=affine_par)
        # for i in self.bn.parameters():
        #     i.requires_grad = False
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(256, num_class, kernel_size=1, stride=1)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.upsample(x)

        return x


class DeepLab(nn.Module):
    def __init__(self, input_channels, dilations):
        super(DeepLab, self).__init__()
        self.tem_module = ResNet(input_channels, dilations)
        self.hum_module = ResNet(input_channels, dilations)
        self.x_wind_module = ResNet(input_channels, dilations)
        self.y_wind_module = ResNet(input_channels, dilations)
        self.multiply = Fusion()
        self.aspp = ASPP(2048)

    def forward(self, x):
        tem_features, tem_features_low = self.tem_module(x[0])
        hum_features, hum_features_low = self.hum_module(x[1])
        x_wind_features, x_wind_features_low = self.x_wind_module(x[2])
        y_wind_features, y_wind_features_low = self.y_wind_module(x[3])
        features = [tem_features, hum_features, x_wind_features, y_wind_features]
        features_low = [tem_features_low, hum_features_low, x_wind_features_low, y_wind_features_low]

        multiplier = self.multiply(features)
        character = 0
        character_low = 0
        for i in range(4):
            m = multiplier[:, i]
            m = m.contiguous()
            m = m.view([-1, 1, 1, 1])
            character += features[i] * m
            character_low += features_low[i] * m

        return self.aspp(character), character_low


class SFNet(nn.Module):
    def __init__(self, input_channels, dilations, num_class):
        super(SFNet, self).__init__()
        self.deeplab = DeepLab(input_channels, dilations)
        self.classifier = Classify(num_class)
        self.conv_reduce = Conv_BN_ReLU(256, 48, kernel_size=1, stride=1)
        self.conv_reduce_next = Conv_BN_ReLU(256, 48, kernel_size=1, stride=1)
        self.conv_cat_1 = Conv_BN_ReLU(256+48, 256, kernel_size=3, stride=1, padding=1)
        self.conv_cat_2 = Conv_BN_ReLU(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_cat_1_next = Conv_BN_ReLU(256 + 48, 256, kernel_size=3, stride=1, padding=1)
        self.conv_cat_2_next = Conv_BN_ReLU(256, 256, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, x):
        x = x.transpose(0, 1)

        features, features_low = self.deeplab(x[0:4])
        features_next, features_next_low = self.deeplab(x[4:8])

        features_low = self.conv_reduce(features_low)
        features_next_low = self.conv_reduce_next(features_next_low)

        features = self.upsample(features)

        features_next = self.upsample(features_next)

        features_ = torch.cat([features, features_low], dim=1)
        features_ = self.conv_cat_1(features_)
        features_ = self.conv_cat_2(features_)
        features_next_ = torch.cat([features_next, features_next_low], dim=1)
        features_next_ = self.conv_cat_1_next(features_next_)
        features_next_ = self.conv_cat_2_next(features_next_)

        pred = self.classifier(torch.cat([features_, features_next_], dim=1))

        return pred
