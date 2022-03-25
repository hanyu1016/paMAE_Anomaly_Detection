from operator import truediv
import sys
sys.path.append("..")
from data_loader_new import *
from model import *
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm, tqdm_notebook

import skimage
from skimage.morphology import square, disk
from skimage import morphology
from skimage import measure

import torch.nn.functional as F
import numbers
from ssim_loss import *
import models
from models import dist_model as dm

from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
#%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def min_max_norm(image, top_perc=0):
    if top_perc == 0 :
        a_min = np.min(image.flatten())
        a_max = np.max(image.flatten())
    else:
        a_min = np.percentile(image.flatten(), q=top_perc)
        a_max = np.percentile(image.flatten(), q=100 - top_perc)
    
    
    return (image - a_min) / (a_max - a_min)

#---------------------------------------------
use_gpu = True         # Whether to use GPU
spatial = True         # Return a spatial map of perceptual distance.
# Linearly calibrated models (LPIPS)
lpips = models.PerceptualLoss(model='net-lin', net='squeeze', use_gpu=use_gpu, spatial=spatial)
# lpips2 = models.PerceptualLoss(model='net-lin', net='squeeze', use_gpu=use_gpu, spatial=False)

#------------------------

model = GeneratorClassConditional(c_dim=3, conv_dim = 32, norm="cbatch",topk=32, no_memory = False, z_dim=512, mem_dim=64, n_downsample=5, num_cls=15)
model.load_weights("checkpoints/2022-01-01 15_28_55.932060 cable_0.1/weights/gen_mae_cls_013000.pth") # +- a few :P
if torch.cuda.is_available():
    model.cuda()
#------------------------

class_subset = "wood"
batch_size = 1
data_mode = "val"
data_path = "./dataset/JET_C+R_Classification/"

#data_loader = get_loader(data_path, dataset="MVTec", shot="all", image_size=[256,256,3],
#                        mode=data_mode, augment=False, shuffle=False, batch_size=batch_size, num_workers=4, subset=class_subset, percent_defect=-1)

# data_loader = get_loader(data_path, dataset='Jet',  image_size=[256,256,3],
#                         mode="val-test", augment=False, shuffle=False, batch_size=batch_size, num_workers=4, subset=class_subset, grayscale=False)
#Hanyu
data_loader = get_loader(data_path, dataset='Jet',  image_size=[256,256,3],
                        mode="test", augment=False, shuffle=False, batch_size=batch_size, num_workers=4, subset=class_subset, grayscale=False)
#Hanyu
print(len(data_loader.dataset.data_list))
predictions = []
#for j, (image, cls_, im_path,_) in enumerate(tqdm_notebook(data_loader)):
#for image, cls_, im_path,_ in enumerate(data_loader):
for batch_index, _data in enumerate(data_loader):
    print(batch_index)
#    if batch_index<500:
#        continue
    #print(len(_data))
    image, cls_, im_path,_=_data[0],_data[1],_data[2],_data[3]
    image = image.cuda()
    model.eval()
    with torch.no_grad():
        rec,w,_,z,z_hat = model(image, cls_)
        
        lpips_err = lpips.forward(image,rec).squeeze(1)
    
        rec_err = torch.mean(torch.abs(rec - image)**2, dim=1)
        rec_err_avg = F.avg_pool2d(rec_err, kernel_size=7, stride=1, padding=3)
        
        error_map = lpips_err ** 2 * rec_err
        
        error_map_avg = lpips_err ** 2 * rec_err_avg
        
    
        for i in range(image.size(0)):
            is_defect = int("good" not in im_path[i])
            
            err_map = error_map[i].cpu().numpy()
            err_map_avg = error_map_avg[i].cpu().numpy()
        
            predictions.append([cls_[i].item(), is_defect, np.mean(err_map), np.max(err_map), np.mean(err_map_avg), np.max(err_map_avg), np.max(lpips_err[i].cpu().numpy()**2)])
        
        img_name=im_path[0]
        while True:
            delete_pos=img_name.find('/')
            if delete_pos<0:
                break
            img_name=img_name[delete_pos+1:]
        delete_pos=img_name.find('.')
        save_img_name=img_name[:delete_pos]
        
        plt.figure(figsize=(20,20))
        plt.subplot(131)
        plt.imshow((image[0].cpu().permute(1,2,0)+1) / 2)
        plt.subplot(132)
        plt.imshow((rec[0].cpu().permute(1,2,0)+1) / 2)
        plt.subplot(133)
        plt.imshow(lpips_err[0].cpu()**2)
        plt.savefig('perceptual_attention_image/OK/'+save_img_name+str(batch_index)+'.jpg')
        #break
#--------------------------------------------------------------------


#from PIL import Image
#pil_image=Image.fromarray((lpips_err[1].cpu()**2).numpy())
#pil_image.show()``


#import cv2
#print(type((lpips_err[1].cpu()**2)))
#print((lpips_err[1].cpu()**2))
#print((lpips_err[1].cpu()**2).shape)
#print(type((lpips_err[1].cpu()**2).numpy()))

#cv2.imwrite((lpips_err[1].cpu()**2).detach().numpy(),'test.png')

#plt.savefig('test.png')