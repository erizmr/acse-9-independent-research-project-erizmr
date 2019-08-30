import torch
import torch.nn as nn
import torch.nn.functional as F

def predict_flow(in_planes):
    """Calculate the current flow inference using convolution transpose"""
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=False)

def deconv(in_planes, out_planes):
    """Calculate the flow inference for next level using convolution transpose with ReLU"""
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1,inplace=True)
    )

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    """Basic convotion manipulation with or without batchnorm"""
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )

def crop(input, target):
    """Match the dimension of input and target dimension"""
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]

def RMSE(input_flow, target_flow, mean=True):
    """Root mean square error"""
    RMSE_err = torch.norm(target_flow-input_flow,2,1)
    batch_size = RMSE_err.size(0)
    if mean:
        return RMSE_err.mean()
    else:
        return RMSE_err.sum()/batch_size

def upRMSE(output, target):
    """Root mean square error with upsampling"""
    b, _, h, w = target.size()
    upsampled_output = F.interpolate(output, (h,w), mode='bilinear', align_corners=False)
    return RMSE(upsampled_output, target, mean=True)
