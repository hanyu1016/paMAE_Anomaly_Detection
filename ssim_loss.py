import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def gaussian(window_size, sigma):
    x = torch.arange(window_size).float()
    gauss = torch.exp(-0.5*(x - window_size // 2) ** 2   / sigma ** 2)
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.transpose(1,0)).float().unsqueeze(0).unsqueeze(0)
    
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()

    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True, out_parts = False):


    img1 = F.pad(img1, (window_size // 2,)*4, mode='reflect')
    img2 = F.pad(img2, (window_size // 2,)*4, mode='reflect')

    mu1 = F.conv2d(img1, window, padding = 0, groups = channel)
    mu2 = F.conv2d(img2, window, padding = 0, groups = channel)

    mu1_sq = mu1**2
    mu2_sq = mu2**2

    mu1_mu2 = mu1*mu2

    # var(x) = E(x^2) - E(x)^2
    sigma1_sq = F.conv2d(img1**2, window, padding = 0, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2**2, window, padding = 0, groups = channel) - mu2_sq
    
    sigma1_sq = sigma1_sq.clamp(min=0)
    sigma2_sq = sigma2_sq.clamp(min=0)

    sigma_1 = torch.sqrt(sigma1_sq)
    sigma_2 = torch.sqrt(sigma2_sq)


    sigma12 = F.conv2d(img1*img2, window, padding = 0, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2
    C3 = C2 / 2
    # C1 = 0.01
    # C2 = 0.03

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if out_parts:
        luminance = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1) 
        contrast = (2 * sigma_1 * sigma_2 + C2) / (sigma1_sq + sigma2_sq + C2)
        structure = (sigma12 + C3) /  (sigma_1 * sigma_2 + C3)
        if size_average:
            return ssim_map.mean(),  luminance.mean(), contrast.mean(), structure.mean() 
        else:
            return ssim_map, luminance, contrast, structure
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map

class SSIM(nn.Module):
    def __init__(self, window_size = 11, size_average = True, out_parts = False):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.out_parts = out_parts

    def forward(self, img1, img2):
        N, channel, H, W = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average,  out_parts = self.out_parts)

def ssim(img1, img2, window_size = 11, size_average = True, out_parts = False):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average, out_parts = out_parts)