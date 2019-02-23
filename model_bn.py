import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16



class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = list(vgg16(pretrained = True).features)[:26]
        self.features = nn.ModuleList(features).eval()
        
    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {1, 6, 11, 18, 25}:
                results.append(x)
        return results


class Resblock(nn.Module):
    def __init__(self, in_channels):
        super(Resblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(in_channels)
        self.conv3 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x_in):
        x = self.conv1_bn(self.conv1(x_in))
        x = self.relu(x)
        x = self.conv2_bn(self.conv2(x))
        x = self.relu(x)
        x = self.conv3_bn(self.conv3(x))
        x = torch.add(x, x_in)
        return x


class Encoding_network(nn.Module):
    def __init__(self):
        super(Encoding_network, self).__init__()
        self.conv1 = nn.Conv2d(512, 512, 1)
        self.resblock1 = Resblock(512)

        self.conv2 = nn.Conv2d(1024, 512, 1)
        self.resblock2 = Resblock(512)

        self.conv3 = nn.Conv2d(768, 256, 1)
        self.resblock3 = Resblock(256)

        self.conv4 = nn.Conv2d(384, 128, 1)
        self.resblock4 = Resblock(128)

        self.conv5 = nn.Conv2d(192, 128, 1)
        self.resblock5 = Resblock(128)

        self.out_conv = nn.Conv2d(128, 64, 1)

    def forward(self, vgg):
        x = self.conv1(vgg[4])
        x = self.resblock1(x)
        x = F.interpolate(x, scale_factor=2)
        x = torch.cat([x, vgg[3]], dim=1)

        x = self.conv2(x)
        x = self.resblock2(x)
        x = F.interpolate(x, scale_factor=2)
        x = torch.cat([x, vgg[2]], dim=1)

        x = self.conv3(x)
        x = self.resblock3(x)
        x = F.interpolate(x, scale_factor=2)
        x = torch.cat([x, vgg[1]], dim=1)

        x = self.conv4(x)
        x = self.resblock4(x)
        x = F.interpolate(x, scale_factor=2)
        x = torch.cat([x, vgg[0]], dim=1)

        x = self.conv5(x)
        x = self.resblock5(x)
        x = self.out_conv(x)
        return x


class Decoding_network(nn.Module):
    def __init__(self):
        super(Decoding_network, self).__init__()
        self.conv1_1 = nn.Conv2d(512, 64, 1)
        self.resblock1 = Resblock(64)
        self.conv1_2 = nn.Conv2d(129, 64, 1)

        self.conv2_1 = nn.Conv2d(512, 64, 1)
        self.resblock2 = Resblock(64)
        self.conv2_2 = nn.Conv2d(129, 64, 1)

        self.conv3_1 = nn.Conv2d(256, 64, 1)
        self.resblock3 = Resblock(64)
        self.conv3_2 = nn.Conv2d(129, 64, 1)

        self.conv4_1 = nn.Conv2d(128, 64, 1)
        self.resblock4 = Resblock(64)
        self.conv4_2 = nn.Conv2d(129, 64, 1)

        self.resblock5 = Resblock(64)
        self.conv5_2 = nn.Conv2d(129, 64, 1)

        self.out_conv = nn.Conv2d(64, 1, 1)

    def forward(self, vgg, enc, cor):
        d_cor = F.interpolate(cor, size=(16, 16))
        d_texture = F.interpolate(enc, size=(16, 16))
        d_v = self.conv1_1(vgg[4])
        x = torch.cat([d_v, d_cor, d_texture], dim=1)
        # print(x.shape, d_v.shape, d_cor.shape, d_texture.shape)
        x = self.conv1_2(x)
        x = self.resblock1(x)

        d_cor = F.interpolate(cor, size=(32, 32))
        x = F.interpolate(x, size=(32, 32))
        d_v = self.conv2_1(vgg[3])
        x = torch.cat([d_v, d_cor, x], dim=1)
        x = self.conv2_2(x)
        x = self.resblock2(x)

        d_cor = F.interpolate(cor, size=(64, 64))
        x = F.interpolate(x, size=(64, 64))
        d_v = self.conv3_1(vgg[2])
        x = torch.cat([d_v, d_cor, x], dim=1)
        x = self.conv3_2(x)
        x = self.resblock3(x)

        d_cor = F.interpolate(cor, size=(128, 128))
        x = F.interpolate(x, size=(128, 128))
        d_v = self.conv4_1(vgg[1])
        x = torch.cat([d_v, d_cor, x], dim=1)
        x = self.conv4_2(x)
        x = self.resblock4(x)

        x = F.interpolate(x, size=(256, 256))
        x = torch.cat([vgg[0], cor, x], dim=1)
        x = self.conv5_2(x)
        x = self.resblock5(x)

        x = self.out_conv(x)
        return x


class Texture_model(nn.Module):
    def __init__(self):
        super(Texture_model, self).__init__()
        self.vgg = Vgg16()
        self.enc = Encoding_network()
        self.enc_ref = Encoding_network()
        self.dec = Decoding_network()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_ref):
        vgg = self.vgg(x)
        enc = self.enc(vgg)
        x = F.normalize(enc)

        x_ref = self.vgg(x_ref)
        x_ref = self.enc_ref(x_ref)
        x_ref = F.normalize(x_ref)

        x = torch.unbind(x)
        x_ref = torch.unbind(x_ref)

        conv_res = []
        for x_t, x_ref_t in zip(x, x_ref):
            x_t, x_ref_t = x_t.unsqueeze(0), x_ref_t.unsqueeze(0)
            res_t = F.conv2d(x_t, x_ref_t, padding=32)
            conv_res.append(torch.squeeze(res_t, dim=0))
        cor = torch.stack(conv_res)

        cor = 1 - cor
        cor = F.interpolate(cor, size=(256, 256))
        # print(x.shape, x_ref.shape, cor.shape)

        out = self.dec(vgg, enc, cor)

        out = self.sigmoid(out)
        return out