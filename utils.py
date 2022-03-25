import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
from scipy.stats import norm
import cv2
from torch.nn import init
import cv2
import h5py 
from torchvision import transforms
import shutil
from PIL import Image
from shutil import copy2

def convert_to_image(inp, mode=None):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((0, 2, 3, 1))
    if mode == "imagenet":
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    else:
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])

    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

def denorm_image(inp, mode=""):
    """Convert a Tensor to numpy image."""
    if mode == "imagenet":
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
    else:
        mean = 0.5
        std = 0.5

    inp = std * inp + mean
    inp = torch.clamp(inp, min=0, max=1)
    return inp


### PyTorch ###
def print_network(model, name):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print(name)
    print("The number of parameters: {}".format(num_params))

def load_network(model, w_path):
    model.load_state_dict(torch.load(w_path, map_location=lambda storage, loc: storage))
    return model

def save_network(model, w_path):
    torch.save(model.state_dict(), w_path)
    
### ======= ###

### OpenCV ###
def to_bgr(image):
    (r,g,b) = cv2.split(image)
    return cv2.merge([b,g,r])

def to_rgb(image):
    (b,g,r) = cv2.split(image)
    return cv2.merge([r,g,b])

def show_image(image, title="Image"):
    # (r,g,b) = cv2.split(image)
    # img = cv2.merge([b,g,r])
    img = image.copy()
    cv2.imshow(title,img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def to_numpy(x):
    if x.is_cuda:
        return np.squeeze(x.cpu().data.numpy())
    return np.squeeze(x.data.numpy())
def to_image(x):
    return np.transpose(to_numpy(denorm(x)),[1,2,0])
def batch_to_list(x):
    N = x.shape[0]
    xs = []
    for i in range(N):
        xs.append(x[i])
    return xs
def save_image_numpy(path, x):
    (r,g,b) = cv2.split(x)
    x = cv2.merge([b,g,r])
    
    x = (x*255).astype("uint8")
    cv2.imwrite(path, x)
def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def mkdir(directory):
    dirs = directory.split("/")
    if directory[:3] == "D:/" or directory[:3]=="C:/":
        print(directory)
        directory = ""
    else:
        directory = "./"
    for d in dirs:
        if d == "." or d == "": continue
        directory += d+"/"
        if not os.path.exists(directory):
            os.makedirs(directory)

def stack_images(images, repeat, h_padding = 0, v_padding = 0, orientation="horizontal"):
    row_images = []
    col_images = []

    height, width = (images[0].shape[0], images[0].shape[1])
    try:
        channel = images[0].shape[2]
    except:
        channel = 1
    
    if orientation == "horizontal":
        v_pad = np.ones([height, v_padding,channel]) 
        w = (width + v_padding)*repeat
        h_pad = np.ones([h_padding,w,channel])
    else:
        h = (height + h_padding)*repeat
        v_pad = np.ones([h, v_padding,channel])
        h_pad = np.ones([h_padding,width,channel])
        
    for i, img in enumerate(images):
        if orientation == "horizontal":
            col_images.append(img)
            col_images.append(v_pad)
            if (i+1) % repeat == 0:
                row_images.append(np.concatenate(col_images,axis=1))
                row_images.append(h_pad)
                col_images = []
        else:
            row_images.append(img)
            row_images.append(h_pad)
            if (i+1) % repeat == 0:
                col_images.append(np.concatenate(row_images,axis=0))
                col_images.append(v_pad)
                row_images = []

    if orientation == "horizontal":
        result = np.concatenate(row_images, axis=0)
    else:
        result = np.concatenate(col_images, axis=1)

    
    if h_padding > 0:
        result = result[:-h_padding, :]
    if v_padding > 0:
        result = result[:, :-v_padding]
    # Trim the paddings
    return result
    
def to_var(x, requires_grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    x.requires_grad_(requires_grad)
    return x

def sample_images(generator, path, x, c_original, z):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    indexes = Image.open("sample_index.png")
    shit = []
    for i in range(16):
        shit.append(transform(indexes.crop((0,i*96,96, i*96+96))).unsqueeze(0))

    shit = torch.cat(shit,dim=0)    
    fake_image_list = [shit, x]
    c_list = generate_c_list(c_original)


    for c in c_list:
        enc, cat, dec, _  = generator(x, c, z)
        fake_image_list.append(dec.data.cpu())
    fake_image_list[1] = fake_image_list[1].data.cpu()
    fake_images = torch.cat(fake_image_list, dim=3)
    save_image(denorm(fake_images), path, nrow=1, padding=0)    

def generate_c_list(original):
    c_list = [original]
    attributes_list = "Bald,Black_Hair,Blond_Hair,Brown_Hair,Gray_Hair,Bangs,Receding_Hairline,Straight_Hair,Wavy_Hair,Bushy_Eyebrows,Eyeglasses,Mouth_Slightly_Open,Mustache,Smiling,Heavy_Makeup,Pale_Skin,Male".split(",")
    hair_group = ["Black_Hair","Blond_Hair","Brown_Hair","Gray_Hair","Bald"]
    for c in attributes_list:
        tmp_target = toggle_attribute_label(original.clone(), c, 1)
        c_list.append(to_var(tmp_target,requires_grad=False))
        if c not in hair_group:
            tmp_target = toggle_attribute_label(original.clone(), c, 0)
            c_list.append(to_var(tmp_target,requires_grad=False))

    return c_list

def toggle_attribute_label(y_label, attribute, value = 1):
    # Check if names is a list
    AVAILABLE_ATTR = ["Bald","Black_Hair","Blond_Hair","Brown_Hair","Gray_Hair","Bangs","Receding_Hairline",
    "Straight_Hair","Wavy_Hair","Wearing_Hat","Bushy_Eyebrows","Eyeglasses","Mouth_Slightly_Open",
    "Mustache","Smiling","Heavy_Makeup","Pale_Skin","Male"]
    
    hair_group = ["Black_Hair","Blond_Hair","Brown_Hair","Gray_Hair","Bald"]

    attributes_list = "Bald,Black_Hair,Blond_Hair,Brown_Hair,Gray_Hair,Bangs,Receding_Hairline,Straight_Hair,Wavy_Hair,Bushy_Eyebrows,Eyeglasses,Mouth_Slightly_Open,Mustache,Smiling,Heavy_Makeup,Pale_Skin,Male".split(",")

    hair_attr_idx = []
    if attribute in hair_group and value == 1:
        for hair in hair_group:
            if hair in attributes_list:
                h_idx = attributes_list.index(hair)
                y_label[:,h_idx] = 0

    attr_idx = attributes_list.index(attribute)

    y_label[:,attr_idx] = value

    return y_label

def init_weights(net, init_type='normal'):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=0.02)
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=1)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            init.normal(m.weight.data, 1.0, 0.02)
            init.constant(m.bias.data, 0.0)
    return init_func        