from cgi import test
import os
import argparse
from solver import Solver
from data_loader_new import *
from torch.backends import cudnn
from utils import *
from datetime import datetime
import zipfile
import torch
import numpy as np

def zipdir(path, ziph):
    files = os.listdir(path)
    for file in files:
        if file.endswith(".py") or file.endswith("cfg"):
            ziph.write(os.path.join(path, file))
            if file.endswith("cfg"):
                os.remove(file)

        if os.path.isdir(file) and file not in ["checkpoints", "backup", "old"]:
            zipdir(file, ziph)


def save_config(config):
    current_time = str(datetime.now()).replace(":", "_")
    
    save_name = "src_files_{}.{}" 
    with open(save_name.format(current_time, "cfg"), "w") as f:
        for k, v in sorted(args.items()):
            f.write('%s: %s\n' % (str(k), str(v)))

    mkdir("backup")
    zipf = zipfile.ZipFile(os.path.join("backup",save_name.format(current_time+ " {}_{}".format(config.subset,config.percent_defect), "zip")),
                           'w', zipfile.ZIP_DEFLATED)
    zipdir('.', zipf)
    zipf.close()

    return current_time + " {}_{}".format(config.subset,config.percent_defect)

def str2bool(v):
    return v.lower() in ('true')

def create_checkpoint_directories(config, version):
    config.model_save_path = os.path.join(config.checkpoint_path, version, config.model_save_path)
    mkdir(config.model_save_path)
    config.result_save_path = os.path.join(config.checkpoint_path, version, config.result_save_path)
    mkdir(config.result_save_path)
    config.log_dir = os.path.join(config.checkpoint_path, version, config.log_dir)
    mkdir(config.log_dir)


def main(version, config):
    # for fast training
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(config.gpu)


    cudnn.benchmark = True
    if config.mode != "test":
        if config.resume is None:   #代表從頭開始 train
            with open("latest_version.txt", "w") as f:
                f.write(version)
        else:   #代表繼續 train
            with open("latest_version.txt", "r") as f:
                version = f.read()        


    if config.resume_version is not None:
        version = config.resume_version
    print("Version: {}".format(version))
    create_checkpoint_directories(config, version)

    ### Save Config to checkpoint ####
    current_time = str(datetime.now()).replace(":", "_")
    save_name = "src_files_{}.{}"
    with open(os.path.join(config.checkpoint_path, version, save_name.format(current_time, "cfg")), "w") as f:
        for k, v in sorted(args.items()):
            f.write('%s: %s\n' % (str(k), str(v)))
    ####################

    # init data loaders
    data_loaders = {}
    data_loaders["train"] = get_loader(config.data_path, dataset=config.dataset, image_size=config.image_dim,
                        mode="train", augment=True, shuffle=True, batch_size=config.batch_size, num_workers=config.num_workers, subset=config.subset, percent_defect=config.percent_defect, grayscale=config.grayscale)
    
    data_loaders["val"] = get_loader(config.data_path, dataset=config.dataset,  image_size=config.image_dim,
                        mode="val", augment=False, shuffle=False, batch_size=config.batch_size, num_workers=config.num_workers, subset=config.subset, grayscale=config.grayscale)
    
    data_loaders["test"] = get_loader(config.inference_test_path, dataset=config.dataset,  image_size=config.image_dim,
                       mode="test", augment=False, shuffle=False, batch_size=1, num_workers=config.num_workers, subset=config.subset, grayscale=config.grayscale)

    print(config.model_save_path)
    solver = Solver(vars(config), data_loaders)
    print(config.resume) # version 要賦給 Resume?
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()
      

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # training settings
    parser.add_argument('--g_lr', type=float, default=1e-4)
    parser.add_argument('--d_lr', type=float, default=4e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--g_conv_dim', type=int, default=32)
    parser.add_argument('--d_conv_dim', type=int, default=32)
    parser.add_argument('--n_downsample', type=int, default=5)
    parser.add_argument('--norm_type', type=str, default="cbatch")
    parser.add_argument('--subset', type=str, default="cable")
    parser.add_argument('--z_dim', type=int, default=512)
    parser.add_argument('--lambda_rec', type=float, default=10)
    parser.add_argument('--lambda_ssim', type=float, default=10)
    parser.add_argument('--lambda_gan_feat', type=float, default=10)
    parser.add_argument('--lambda_lpips', type=float, default=10)
    parser.add_argument('--lambda_vgg', type=float, default=10)
    parser.add_argument('--lambda_mem_entropy', type=float, default=0.01)
    parser.add_argument('--lambda_feat_compact', type=float, default=1.0)
    parser.add_argument('--lambda_feat_sep', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=1000, help='Set patience of lr scheduler')

    # parser.add_argument('--resume', type=str,
    #                     default='2022-03-09 21_59_18.293081 cable_0.1', help='set which checkpoint (epoch / iter) to resume, if resume version is None then use latest version')
    
    # parser.add_argument('--resume_version', type=str,
    #                     default='2022-01-01 15_28_55.932060 cable_0.1', help='set which version to resume')

    parser.add_argument('--resume', type=str,
                        default=None, help='set which checkpoint (epoch / iter) to resume, if resume version is None then use latest version')
    
    parser.add_argument('--resume_version', type=str,
                        default=None, help='set which version to resume')

    
    parser.add_argument('--mem_dim', type=int, default=64)
    parser.add_argument('--mem_cls', type=str2bool, default=True)
    #parser.add_argument('--num_cls', type=int, default=2)
    parser.add_argument('--num_cls', type=int, default=15)
    parser.add_argument('--topk', type=int, default=3)
    # parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--clip_margin', type=int, default=20)
    parser.add_argument('--percent_defect', type=float, default=0.1)


    parser.add_argument('--with_discriminator', type=str2bool, default=True)
    parser.add_argument('--no_vgg_loss', action='store_true')
    parser.add_argument('--no_memory', action='store_true')
    parser.add_argument('--no_ssim', action='store_true')
    parser.add_argument('--no_lpips_loss', action='store_true')
    parser.add_argument('--no_gan_feat_loss', action='store_true')
    parser.add_argument('--no_memfeat_loss', action='store_true')
    parser.add_argument('--grayscale', action='store_true')

    # dataset info
    parser.add_argument('--dataset', type=str, default='Jet', choices=['MVTec', 'Jet'])
    parser.add_argument('--image_dim', type=list, default=[256,256,3])
    # parser.add_argument('--image_dim', type=list, default=[128,128,3])
    parser.add_argument('--num_workers', type=int, default=10)

    # step size
    parser.add_argument('--counter', type=str, default='iter',
                        choices=['iter', 'epoch'])
    parser.add_argument('--num_iterations', type=int, default=10000000)
    parser.add_argument('--num_epochs', type=int, default=36)
    parser.add_argument('--loss_log_step', type=int, default=500)
    parser.add_argument('--sample_step', type=int, default=500)
    parser.add_argument('--model_save_step', type=int, default=1000)

    # scheduler settings
    parser.add_argument('--warmup', type=str2bool, default=False)
    parser.add_argument('--warmup_step', type=int, default=6)
    parser.add_argument('--sched_milestones', type=list,
                        default=[80000, 100000, 120000])
    parser.add_argument('--sched_gamma', type=float, default=0.1)

    # loss settings
    parser.add_argument('--threshold', type=float, default=0.5)

    # misc
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'adapt', 'test'])
    parser.add_argument('--use_gpu', type=str2bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # path
    # pamae_xiaosean_version_20220125/dataset/JET_C+R_Classification/Training_set/OK/word
    parser.add_argument('--data_path', type=str, default="../dataset/JET_C+R_Classification/")

    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints')
    parser.add_argument('--model_save_path', type=str, default='weights')
    parser.add_argument('--result_save_path', type=str, default='results')
    parser.add_argument('--log_dir', type=str, default='logs')

    # Inference
    parser.add_argument('--inference_model_path', type=str, default='inference_model/gen_mae_cls_latest.pth')
    parser.add_argument('--inference_test_path', type=str, default='TestData')
    parser.add_argument('--inference_result_path', type=str, default='predict_score.txt')
    parser.add_argument('--inference_threshold', type=float, default=0.8)
    

    config = parser.parse_args()

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------') 

    # if config.mode =='train' :
    #     version = save_config(config)  #training 時要用這裡
    #     print("Start Training!")
    # else:
    #     version = '2022-01-01 15_28_55.932060 cable_0.1'    #這裡是test用的
    #     print("Start Testing!")
    
    version = save_config(config)  #training 時要用這裡

    print('-----------------------------------')
    print(version)
    main(version, config)

