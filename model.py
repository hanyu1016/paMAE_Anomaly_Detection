import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
import torch.nn.utils.spectral_norm as spectral_norm
import torchvision
import math

import matplotlib.pyplot as plt

from torch.autograd import Function
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel,self).__init__()
        self.model_name = "model"

    def init_layer_weights(self, gain=0.02):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, gain)
                    nn.init.constant_(m.bias.data, 0.0)


    def save_weights(self, path):
        torch.save({
            'model_state_dict': self.state_dict()
        }, path)

    def load_weights(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])

    def save_checkpoint(self, path, optimizer, iteration):
        torch.save({
            'iteration': iteration,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(path, "{}_{}.pth".format(self.model_name,str(iteration).zfill(6))))

        torch.save({
            'iteration': iteration,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(path, "{}_{}.pth".format(self.model_name,"latest")))

    def load_checkpoint(self, path, resume, optimizer):
        checkpoint = torch.load(os.path.join(path, "{}_{}.pth".format(self.model_name, resume.zfill(6))))
        self.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint["iteration"]

    def load_inference_model(self, path, optimizer):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint["iteration"]

    def load_base_weights(self, path):
        self.load_weights(path)


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out, norm="batch", num_cls=10, guide_dim=512):
        super(ResidualBlock, self).__init__()
        self.learned_shortcut = (dim_in != dim_out)
        self.norm = norm
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=False)


        layers = []
        layers.append(nn.ReflectionPad2d(1))
        layers.append(spectral_norm(nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=0, bias=False)))

        if norm == "batch":
            layers.append(nn.BatchNorm2d(dim_out, affine=True))
        elif norm == "instance":
            layers.append(nn.InstanceNorm2d(dim_out, affine=False))
        elif norm == "cbatch":
            layers.append(ConditionalBatchNorm(num_cls, dim_out))
        elif norm == "ginstance":
            layers.append(GuidedInstanceNorm(guide_dim, dim_out))
        layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.ReflectionPad2d(1))
        layers.append(spectral_norm(nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=0, bias=False)))

        if norm == "batch":
            layers.append(nn.BatchNorm2d(dim_out, affine=True))
        elif norm == "instance":
            layers.append(nn.InstanceNorm2d(dim_out, affine=False))
        elif norm == "cbatch":
            layers.append(ConditionalBatchNorm(num_cls, dim_out))

        self.main = nn.Sequential(*layers)

    def shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s

    def forward(self, x, cls=None):

        if (self.norm == "cbatch" or self.norm == "cinstance" or self.norm == "ginstance" ) and cls is not None:
            h = x
            for layer in self.main:
                if isinstance(layer, (ConditionalBatchNorm)):
                    h = layer(h, cls)
                else:
                    h = layer(h)
            main = h

            return self.shortcut(x) + main
        else:
            return self.shortcut(x) + self.main(x)


class ConditionalBatchNorm(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, label_nc, norm_nc):
        super(ConditionalBatchNorm, self).__init__()
        self.label_nc = label_nc
        nhidden = 128

        self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)

        self.mlp_shared = nn.Sequential(
            nn.Linear(label_nc, nhidden),
            nn.ReLU()
        )
        
        self.mlp_gamma = nn.Linear(nhidden, norm_nc)
        self.mlp_beta = nn.Linear(nhidden, norm_nc)

    def forward(self, x, cls_label):
        normalized = self.param_free_norm(x)
        
        one_hot = torch.eye(self.label_nc)[cls_label].cuda()
   
        actv = self.mlp_shared(one_hot)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        out = normalized * (1 + gamma.unsqueeze(-1).unsqueeze(-1)) + beta.unsqueeze(-1).unsqueeze(-1)
        return out 


class GeneratorClassConditional(BaseModel):
    def __init__(self, c_dim=1, conv_dim=64, hole_size=32, topk=-1, guide_dim=512, n_downsample=4, clip_margin=20, z_dim=512, norm="batch", no_memory=False, mem_dim=100, num_cls=10):
        super(GeneratorClassConditional, self).__init__()
        self.hole_size = hole_size
        self.no_memory = no_memory
        self.memory = MemoryClsEuclideanRW(z_dim=z_dim, mem_dim=mem_dim, num_cls=num_cls, topk=topk, clip_margin=clip_margin)
        self.norm = norm
        self.model_name = "gen_mae_cls"

        # pre-processing stage. Outputs a tensor of with spatial size of 64 x 64
        layers = []
        layers.append(nn.ReflectionPad2d(3))
        layers.append(spectral_norm(nn.Conv2d(c_dim, conv_dim, kernel_size=7, stride=1, padding=0, bias=False)))
        if norm == "batch":
            layers.append(nn.BatchNorm2d(conv_dim, affine=True))
        elif norm == "instance":
            layers.append(nn.InstanceNorm2d(conv_dim, affine=False))
        elif norm == "cbatch":
            layers.append(ConditionalBatchNorm(num_cls, conv_dim))

        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(n_downsample):

            out_dim = min(z_dim, curr_dim * 2)
            if i == n_downsample-1:
                out_dim = z_dim
            layers.append(nn.ReflectionPad2d(1))
            layers.append(spectral_norm(nn.Conv2d(curr_dim, out_dim, kernel_size=3, stride=2, padding=0, bias=False)))

            if norm == "batch":
                layers.append(nn.BatchNorm2d(out_dim, affine=True))
            elif norm == "instance":
                layers.append(nn.InstanceNorm2d(out_dim, affine=False))
            elif norm == "cbatch":
                layers.append(ConditionalBatchNorm(num_cls, out_dim))

            
            layers.append(nn.ReLU(inplace=True))
            curr_dim = out_dim

        for i in range(3):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, norm=norm, num_cls=num_cls))

        self.encoder = nn.Sequential(*layers)


        layers = []
        # Bottleneck layers.
        for i in range(3):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, norm=norm, num_cls=num_cls))
        # Up-sampling layers.
        for i in reversed(range(n_downsample)):
            curr_dim = min(z_dim, conv_dim * 2 ** (i+1) )
            if i == n_downsample-1:
                curr_dim = z_dim
            out_dim = min(z_dim, conv_dim * 2 ** (i))

            layers.append(nn.ReflectionPad2d(1))
            layers.append(spectral_norm(nn.Conv2d(curr_dim, out_dim * 4, kernel_size=3, stride=1, padding=0, bias=False)))
            layers.append(nn.PixelShuffle(2))

            if norm == "batch":
                layers.append(nn.BatchNorm2d(out_dim, affine=True))
            elif norm == "instance":
                layers.append(nn.InstanceNorm2d(out_dim, affine=False))
            elif norm == "cbatch":
                layers.append(ConditionalBatchNorm(num_cls, out_dim))

            layers.append(nn.ReLU(inplace=True))
            


        layers.append(nn.ReflectionPad2d(3))
        layers.append(nn.Conv2d(conv_dim, c_dim, kernel_size=7, stride=1, padding=0, bias=False))
        layers.append(nn.Tanh())

        self.decoder = nn.Sequential(*layers)


        self.init_layer_weights()
    def z_from_w(self, w, cls, memory=None, C=512, H=8, W=8):
        return self.memory.z_from_w(w,cls,memory)
    
    def decode_from_z(self,z, cls):
        h = z
        for layer in self.decoder:
            if isinstance(layer, (ConditionalBatchNorm,  ResidualBlock)):
                h = layer(h, cls)
            else:
                h = layer(h)
        dec = h
        return dec
    def forward(self, x, cls= None, memory=None):
        if not self.no_memory:

            if (self.norm == "cbatch") and cls is not None:

                h = x
                for layer in self.encoder:
                    if isinstance(layer, (ConditionalBatchNorm)):
                        h = layer(h, cls)
                    elif isinstance(layer, (ResidualBlock)):
                        h = layer(h, cls)
                    else:
                        h = layer(h)
                enc = h
                # enc = enc + torch.randn(enc.size()).cuda() * 0.02
                z_hat, w_hat, w, log_w = self.memory(enc, cls, memory=memory)
  
                h = z_hat
                for layer in self.decoder:
                    if isinstance(layer, (ConditionalBatchNorm)):
                        h = layer(h, cls)
                    elif isinstance(layer, (ResidualBlock)):
                        h = layer(h, cls)
                    else:
                        h = layer(h)
                dec = h

                return dec, w, log_w, enc, z_hat
            else:
                enc = self.encoder(x)
                z_hat, w_hat, w, log_w = self.memory(enc, cls)
                dec = self.decoder(z_hat)

                return dec, w, log_w
        # elif self.norm == "cbatch":
        #     h = x
        #     for layer in self.encoder:
        #         if isinstance(layer, (ConditionalBatchNorm)):
        #             h = layer(h, cls)
        #         elif isinstance(layer, (ResidualBlock)):
        #             h = layer(h, cls)
        #         else:
        #             h = layer(h)
        #     enc = h
        #     # enc = enc + torch.randn(enc.size()).cuda() * 0.02
        #     for layer in self.decoder:
        #         if isinstance(layer, (ConditionalBatchNorm)):
        #             h = layer(h, cls)
        #         elif isinstance(layer, (ResidualBlock)):
        #             h = layer(h, cls)
        #         else:
        #             h = layer(h)
        #     dec = h

        #     return dec

        else:
            enc = self.encoder(x)
            dec = self.decoder(enc)
            
            return dec


class Discriminator(BaseModel):
    """Discriminator network with PatchGAN."""
    def __init__(self, conv_dim=64, c_dim=1, repeat_num=4, norm="batch"):
        super(Discriminator, self).__init__()
        layers = []

        sublayer = []
        sublayer.append(nn.Conv2d(c_dim, conv_dim, kernel_size=4, stride=2, padding=2))
        sublayer.append(nn.LeakyReLU(0.2))

        layers.append(nn.Sequential(*sublayer))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            out_dim = min(curr_dim * 2, 512)
            stride = 1 if i == repeat_num - 1 else 2

            sublayer = []
            sublayer.append(spectral_norm(nn.Conv2d(curr_dim, out_dim, kernel_size=4, stride=stride, padding=2)))
            
            if norm == "batch":
                sublayer.append(nn.BatchNorm2d(out_dim, affine=True))
            elif norm == "instance":
                sublayer.append(nn.InstanceNorm2d(out_dim, affine=False))
            
            sublayer.append(nn.LeakyReLU(0.2))


            layers.append(nn.Sequential(*sublayer))
            curr_dim = out_dim

        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=4, stride=1, padding=2, bias=False)
        
        self.init_layer_weights()

    def forward(self, x, output_features=False):
        if output_features:
            out = x
            features = []
            for layer in self.main:
                out = layer(out)
                features.append(out)

            out_pred = self.conv1(out)

            return out_pred, features
        else:
            h = self.main(x)
            out_pred = self.conv1(h)
            
            return out_pred

class MemoryClsEuclideanRW(nn.Module):
    def __init__(self, z_dim=256, mem_dim=2000, num_cls=10, topk=-1, clip_margin=20):
        super(MemoryClsEuclideanRW, self).__init__()
        self.z_dim = z_dim
        self.mem_dim = mem_dim
        self.num_cls = num_cls
        self.memory = nn.Parameter(torch.Tensor(1, z_dim, mem_dim, num_cls))
        self.R = nn.Parameter(torch.tensor(0.))
        self.memory_access_mask = torch.ones(1,1,self.mem_dim).cuda()
        self.iter = 0
        self.clip_margin = clip_margin
        # self.focus_beta = nn.Parameter(torch.ones(num_cls))
        self.topk = topk
        # nn.init.normal_(self.memory, 0.0, 0.1)
        nn.init.normal_(self.memory, 0.0, 0.02)
        self.quantile = lambda t, q: t.view(-1).kthvalue(1 + round(float(q) * (t.numel() - 1))).values.item()
    def get_soft_address(self, z, cls, memory=None):
        if memory is None:
            memory = self.memory
        # cls N x K

        # z is (N x C x H x W)
        N, C, H, W = z.size()
        
        # reshape to N x HW x C x 1
        z_reshaped = z.view(N, C, H*W, 1).permute(0, 2, 1, 3)
        
        # 1 x C x M x K => 1 x C x M x N
        memory_cls = memory[:,:,:,cls]
        # N x 1 x C x M
        memory_cls = memory_cls.permute(3, 0, 1, 2)


        # N x HW x C x 1 - N x 1 x C x M => N x HW x C x M => N x HW x M
        similarity = -torch.sum((z_reshaped - memory_cls)**2, dim=2) 
        # focused_similarity = self.focus_beta[cls][:,None,None]*similarity
        focused_similarity = similarity

        log_similarity = F.log_softmax(focused_similarity, dim=2)
        similarity_normed = F.softmax(focused_similarity, dim=2)

        return similarity_normed, log_similarity

    def update_memory_loss(self, z, w, cls):
        # cls N x K

        # z is (N x C x H x W)
        N, C, H, W = z.size()
        
        # reshape to N x HW x C x 1
        z_reshaped = z.view(N, C, H*W, 1).permute(0, 2, 1, 3)
        
        # 1 x C x M x K => 1 x C x M x N
        memory_cls = self.memory[:,:,:,cls]
        # N x 1 x C x M
        memory_cls = memory_cls.permute(3, 0, 1, 2)

        # N x HW x C x 1 - N x 1 x C x M => N x HW x C x M => N x HW x M
        similarity = w

        # import pdb; pdb.set_trace()
        # N x HW x 2
        _, idx_m = torch.topk(similarity, k=2, dim=2)

        triplet_loss = nn.TripletMarginLoss(margin=1.0)

        mem_view = memory_cls.expand(N, H*W, C, self.mem_dim).reshape(N*H*W, C, self.mem_dim)
        mem_pos = mem_view[torch.arange(N*H*W),:,idx_m[:,:,0].view(-1)].view(N,H*W, C)
        mem_neg = mem_view[torch.arange(N*H*W),:,idx_m[:,:,1].view(-1)].view(N,H*W, C)

        #N x HW x C -> N x HW x 1 -> NHW x 1 
        z_dist = torch.sum((z_reshaped.squeeze() - mem_pos)**2, dim=2, keepdim=True).view(N*H*W, 1)
        
        loss_feat_compact = torch.mean((z_reshaped.squeeze() - mem_pos)**2)
        loss_feat_sep = triplet_loss(z_reshaped.squeeze(), mem_pos ,mem_neg)

        return loss_feat_compact, loss_feat_sep
    
    def z_from_w(self, w, cls, memory=None, C=512, H=8, W=8):
        if memory is None:
            memory = self.memory
        
        w_hat = self.topk_memory(w, k=self.topk)
        
        
        # 1 x C x M x K => C x M x N => N x C x M
        memory_cls = memory[0, :, :, cls].permute(2, 0, 1)

        # N x C x M 
        z_hat = torch.bmm(memory_cls, w_hat.permute(0, 2, 1))

        z_hat_reshaped = z_hat.view(w_hat.size(0), C, H, W)

        return z_hat_reshaped

    def topk_memory(self, w, k=3):
        if k == -1:
            return w
        N, HW, M = w.size()

        mask = torch.zeros_like(w)
        # N x HW x M
        _, idx_m = torch.topk(w, k=k, dim=2)
        w_hat = w * mask.scatter_(dim=2, index=idx_m, value = 1.0)
        w_hat = w_hat / w_hat.sum(dim=2, keepdim = True)


        return w_hat
    def forward(self, z, cls, memory=None):
        if memory is None:
            memory = self.memory
        N, C, H, W = z.size()
        # N x HW x M 
        w, log_w = self.get_soft_address(z, cls, memory)

        # w_hat = self.hard_shrinkage(w)
        w_hat = w

        # w_hat = self.topk_memory(w_hat, k=self.topk)

        # 1 x C x M x K => C x M x N => N x C x M
        memory_cls = memory[0, :, :, cls].permute(2, 0, 1)


        # N x C x M 
        z_hat = torch.bmm(memory_cls, w_hat.permute(0, 2, 1))

        z_hat_reshaped = z_hat.view(N, C, H, W)

        return z_hat_reshaped, w_hat, w, log_w

# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [10.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
