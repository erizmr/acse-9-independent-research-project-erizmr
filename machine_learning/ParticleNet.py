import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from net_tools import *
from collections import OrderedDict

class ParticleNet(nn.Module):
    expansion = 1
    def __init__(self, batchNorm=True):
        super(ParticleNet, self).__init__()

        self.batchNorm = batchNorm
        self.conv1   = conv(self.batchNorm,   2,   64, kernel_size=7, stride=2)
        self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)
        self.conv3   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256,  256)
        self.conv4   = conv(self.batchNorm, 256,  512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512,  512)
        self.conv5   = conv(self.batchNorm, 512,  512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512,  512)
        self.conv6   = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm,1024, 1024)

        self.deconv5 = deconv(1024,512)
        self.deconv4 = deconv(1026,256)
        self.deconv3 = deconv(770,128)
        self.deconv2 = deconv(386,64)
        self.deconv1 = deconv(194,2)
        self.deconv0 = deconv(68,2)
        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)
        self.predict_flow1 = predict_flow(68)
        self.predict_flow0 = predict_flow(6)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = crop(self.upsampled_flow6_to_5(flow6), out_conv5)
        out_deconv5 = crop(self.deconv5(out_conv6), out_conv5)

        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        flow5       = self.predict_flow5(concat5)
        flow5_up    = crop(self.upsampled_flow5_to_4(flow5), out_conv4)
        out_deconv4 = crop(self.deconv4(concat5), out_conv4)

        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        flow4       = self.predict_flow4(concat4)
        flow4_up    = crop(self.upsampled_flow4_to_3(flow4), out_conv3)
        out_deconv3 = crop(self.deconv3(concat4), out_conv3)

        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        flow3       = self.predict_flow3(concat3)
        flow3_up    = crop(self.upsampled_flow3_to_2(flow3), out_conv2)
        out_deconv2 = crop(self.deconv2(concat3), out_conv2)

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        flow2 = self.predict_flow2(concat2)
        flow2_up = crop(self.upsampled_flow2_to_1(flow2), out_conv1)
        out_deconv1 = crop(self.deconv1(concat2), out_conv1)

        concat1 = torch.cat((out_conv1,out_deconv1,flow2_up),1)
        flow1 = self.predict_flow1(concat1)
        flow1_up = crop(self.upsampled_flow1_to_0(flow1), x)
        out_deconv0 = crop(self.deconv0(concat1), x)

        concat0 = torch.cat((x,out_deconv0,flow1_up),1)
        flow0 = self.predict_flow0(concat0)


        if self.training:
            return flow0,flow1,flow2,flow3,flow4,flow5,flow6
        else:
            return flow0

def CE(network_output, target_flow, weights=None):
    """Composite error defined for training"""
    def scale(output, target):
        b, _, h, w = output.size()
        target_scaled = F.interpolate(target, (h, w), mode='area')
        return RMSE(output, target_scaled, mean=False)
    if type(network_output) not in [tuple, list]:
        network_output = [network_output]
    if weights is None:
        weights = [0.005, 0.01, 0.01, 0.02, 0.04, 0.08, 0.32]
    assert(len(weights) == len(network_output))

    loss = 0
    for output, weight in zip(network_output, weights):
        loss += weight * scale(output, target_flow)
    return loss
