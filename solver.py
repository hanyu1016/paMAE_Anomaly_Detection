import os
import time
import datetime
import pickle
from unittest import result
from importlib_metadata import version

#import colorama
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
import numpy as np
from PIL import Image, ImageDraw
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
import pandas as pd 

import models
from model import *
from utils import *
from ssim_loss import *


class Solver(object):
    # """Solver for training and testing StarGAN."""
    DEFAULTS = {}
    def __init__(self, config, data_loaders):
        # Initialize configurations
        self.__dict__.update(Solver.DEFAULTS, **config)

        # Data_loaders is a dictionary with key "train", "test"
        self.data_loaders = data_loaders

        # Build the model and tensorboard.
        self.init_model()
        if self.use_tensorboard:
            self.build_tensorboard()

        self.memory_access_tracker = {}

    def init_model(self):
        # Initialize Model
        c_dim = 3
        if self.grayscale:
            c_dim = 1
        self.generator = GeneratorClassConditional(c_dim=c_dim, topk=self.topk, conv_dim = self.g_conv_dim, norm=self.norm_type, no_memory = self.no_memory, z_dim=self.z_dim, mem_dim=self.mem_dim, n_downsample=self.n_downsample, num_cls=self.num_cls, clip_margin=self.clip_margin)
        
        if self.with_discriminator:
            self.discriminator = Discriminator(c_dim=c_dim, conv_dim= self.d_conv_dim, repeat_num=4, norm=self.norm_type)

        # Initialize Optimizers
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), self.g_lr, betas=[0, 0.9])
        if self.with_discriminator:
            self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), self.d_lr, betas=[0, 0.9])

        print_network(self.generator, 'Generator')
        if self.with_discriminator:
            print_network(self.discriminator, 'Discriminator')

        if not self.no_vgg_loss:
            self.vgg_loss = VGGLoss()

        self.lpips_loss = models.PerceptualLoss(model='net-lin', net='squeeze', use_gpu=True, spatial=True)
        
        if torch.cuda.is_available():
            self.generator.cuda()
            if self.with_discriminator:
                self.discriminator.cuda()
                
    def test(self):

        # assert self.resume is not None # True 時會執行
        # print(self.model_save_path)
        # print(self.version)

        # start_iters = self.generator.load_checkpoint(self.model_save_path, 
        #                                         self.resume, 
        #                                         self.g_optimizer)

        self.generator.load_inference_model(self.inference_model_path,self.g_optimizer)

        # In test mode, we don't consider discriminator
        # if self.with_discriminator:
        #     self.discriminator.load_checkpoint(self.model_save_path, 
        #                                             self.resume, 
        #                                             self.d_optimizer)
        # print("Loading from checkpoint: {}".format(self.resume))
        # avg_loss = self.sample_validation(self.generator, self.data_loaders["test"], 
                #                                      prefix="test",train_iters=i+1, val_num_iters=2)
        '''
        avg_loss = self.get_std_by_validation(self.generator, self.data_loaders["val"], 
                                            prefix="val")
        log = "#"*10 + " Validation " + "#"*10 + "\n"
        for tag, value in sorted(avg_loss.items()):
            log += "{}: {:.4f}\n".format(tag, value)
        log += "#"*32
        print(log)
        '''

        '''
        # print("Top 100 loss images in training set")
        avg_loss = self.get_std_by_validation(self.generator, self.data_loaders["val-test"], 
                                            prefix="val")

        #self.lpips_loss = models.PerceptualLoss(model='net-lin', net='squeeze', use_gpu=True, spatial=True)
        #loss_lpips = self.lambda_lpips * self.lpips_loss.forward(input_image.detach(),reconstructed).mean()

        log = "#"*10 + " Validation_defect " + "#"*10 + "\n"
        for tag, value in sorted(avg_loss.items()):
            log += "{}: {:.4f}\n".format(tag, value)
        log += "#"*32
        print(log)
        '''

        # Select type: avg_loss["ssim_mean"], avg_loss["ssim_std"], avg_loss["rec_mean"],
        #              avg_loss["rec_std"], avg_loss["total_mean"], avg_loss["total_std"]        
        
        # Detect training set
        #anomaly_cnt = self.detect_anomaly(self.generator, self.data_loaders["train"], mean=0,\
        #                                  std=0, eval_type="rec", out_filename="training_error")
        
        # Detect val set
        #anomaly_cnt = self.detect_anomaly(self.generator, self.data_loaders["val"], mean=0,\
        #                                  std=0, eval_type="rec", out_filename="val_error")    
        
        '''
        # Detect val-defect
        anomaly_cnt = self.detect_anomaly(self.generator, self.data_loaders["val-test"], 
                            mean=0, std=0, eval_type="rec", out_filename="val_defect")
        '''
        
        # lpip_loss_list=self.get_test_perceptual_loss(self.generator, self.data_loaders["val-test"])
        print(len(self.data_loaders["test"].dataset))
        lpip_loss_list=self.get_test_perceptual_loss(self.generator, self.data_loaders["test"])

        
        print(type(lpip_loss_list))
        print(len(lpip_loss_list))

        min_loss=min(lpip_loss_list)
        max_loss=max(lpip_loss_list)
        
        print('max_loss:',max_loss)
        print('min_loss:',min_loss)
        
        file1=open(self.inference_result_path,'w')
        for i, (input_image, class_id, im_id, affine_theta) in enumerate(self.data_loaders["test"]):
            if lpip_loss_list[i]>self.inference_threshold:
                ok_ng=1
            else:
                ok_ng=0
            file1.write(im_id[0]+'\t'+str(float(lpip_loss_list[i]))+'\t'+str(ok_ng)+'\n')
        file1.close()

        # anomaly_cnt = self.detect_anomaly(self.generator, self.data_loaders["val-test"], 
                            # mean=avg_loss["rec_mean"], std=avg_loss["rec_std"], eval_type="rec", out_filename="val_defect.csv")
        
        #anomaly_cnt = self.detect_anomaly(self.generator, self.data_loaders["test"], 
        #                    mean=0, std=0, eval_type="rec", out_filename="test.csv", confusion_mat=False)
        
        # print("anomaly_cnt =", anomaly_cnt)
        

        

    def train(self):
        print("#"*20)
        print("Train Mode")
        print("#"*20)
        start_iters = 0



        if self.resume is not None:
            start_iters = self.generator.load_checkpoint(self.model_save_path, 
                                                    self.resume, 
                                                    self.g_optimizer)
            if self.with_discriminator:
                self.discriminator.load_checkpoint(self.model_save_path, 
                                                        self.resume, 
                                                        self.d_optimizer)
            print("Loading from checkpoint: {}".format(self.resume))

        #print(len(self.data_loaders["train"].dataset.data_list))
        #print(self.data_loaders["train"].batch_size)
        data_iter = iter(self.data_loaders["train"])

        #data_iter = next()
        iters_per_epoch = len(self.data_loaders["train"]) 

        if self.counter == "iter":
            num_epochs = np.ceil(self.num_iterations / iters_per_epoch)
            num_iters = self.num_iterations
        else:
            num_epochs = self.num_epochs
            num_iters = self.num_epochs * iters_per_epoch
        
        print('Start training at iter : {} / {}'.format(start_iters, num_iters))
        start_time = time.time()
        
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.g_optimizer, 'min', patience=self.patience)

        for i in trange(start_iters, num_iters, position=1, leave=True):
            # Print out training information.
            epoch = i // iters_per_epoch

            try:
                input_image, class_id,_, affine_theta = next(data_iter)
            except StopIteration:
                data_iter = iter(self.data_loaders["train"])
                input_image, class_id,_, affine_theta= next(data_iter)
            
            if torch.cuda.is_available():
                input_image = input_image.cuda()
                class_id = class_id.cuda()
                affine_theta = affine_theta.cuda()

            if self.with_discriminator:
                loss_D = self.discriminator_train_step(input_image, class_id, affine_theta)

            loss_G = self.generator_train_step(input_image, class_id, affine_theta, i)


            del input_image
            torch.cuda.empty_cache()

            if (i+1) % self.loss_log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]

                elapsed = time.time() - start_time

                total_time = (num_iters-i) * elapsed/(i+1 - start_iters)

                total_time = str(datetime.timedelta(seconds=total_time))[:-7]
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]

                log = "\nElapsed [{}], Epoch {} - Iteration [{}/{}]\n".format(et, epoch, i+1, num_iters)

                for param_group in self.g_optimizer.param_groups:
                    log += "g_lr:{}\n".format(param_group['lr'])
                if self.with_discriminator:
                    for param_group in self.d_optimizer.param_groups:
                        log += "d_lr:{}\n".format(param_group['lr'])
                
                    for tag, value in sorted(loss_D.items()):
                        log += "{}: {:.4f}\n".format(tag, value)
                for tag, value in sorted(loss_G.items()):
                    log += "{}: {:.4f}\n".format(tag, value)
                
                log += "Elapsed / Total {}/{}\n".format(elapsed,total_time)
                print(log)

                if self.use_tensorboard:
                    if self.with_discriminator:
                        for tag, value in loss_D.items():
                            self.logger.scalar_summary(tag, value, i+1)

                    for tag, value in loss_G.items():
                        self.logger.scalar_summary(tag, value, i+1)

            if (i+1) % self.sample_step == 0:
                # avg_loss = self.sample_validation(self.generator, self.data_loaders["test"], 
                #                                      prefix="test",train_iters=i+1, val_num_iters=2)
                #Hanyu      
                # avg_loss = self.sample_validation(self.generator, self.data_loaders["val"], 
                #                                     prefix="val", train_iters=i+1)
                #Hanyu
                avg_loss = self.sample_validation(self.generator, self.data_loaders["val"], 
                                                    prefix="val", train_iters=i+1)
                log = "#"*10 + " Validation " + "#"*10 + "\n"
                for tag, value in sorted(avg_loss.items()):
                    log += "{}: {:.4f}\n".format(tag, value)
                log += "#"*32
                print(log)
                #Hanyu
                # avg_loss = self.sample_validation(self.generator, self.data_loaders["val-test"], 
                #                                     prefix="val", train_iters=i+1)
                #Hanyu
                self.lr_scheduler.step(avg_loss["val/total"])

                if self.use_tensorboard:
                    for tag, value in avg_loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                self.generator.save_checkpoint(
                    self.model_save_path, self.g_optimizer, i+1)
                if self.with_discriminator:
                    self.discriminator.save_checkpoint(
                        self.model_save_path, self.d_optimizer, i+1)

                print('Saved model checkpoints into {}...'.format(self.model_save_path))



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

    def discriminator_train_step(self, original_image, class_id, affine_theta=None):
        self.d_optimizer.zero_grad()


        N, C, H, W = original_image.size()

        input_image = original_image
        with torch.no_grad():
            if self.no_memory:
                fake_image = self.generator(input_image, class_id)
            else:
                fake_image, w, log_w, z, z_hat = self.generator(input_image, class_id)
                # fake_image = self.warp_image(fake_image, affine_theta[:,:,:,1],img_size=(input_image.size(2),input_image.size(3)))



            fake_image = fake_image.detach()
            fake_image.requires_grad_()
       
        combined = torch.cat([input_image, fake_image], dim=0)


        global_predictions = self.discriminator(combined, output_features=False)
        real_global_pred = global_predictions[:N]
        fake_global_pred = global_predictions[N:]

        minval = torch.min(real_global_pred - 1, torch.zeros_like(real_global_pred))
        loss_global_real = -torch.mean(minval)
 
        minval = torch.min(-fake_global_pred - 1, torch.zeros_like(fake_global_pred))
        loss_global_fake = -torch.mean(minval)

        total_loss = loss_global_real + loss_global_fake

        total_loss.backward()
        self.d_optimizer.step()




        loss = {}
        loss["train/D/loss_global_fake"] = loss_global_fake.data.item()
        loss["train/D/loss_global_real"] = loss_global_real.data.item()
        loss["train/D/total_loss"] = total_loss.data.item()

        return loss

    def generator_train_step(self, original_image, class_id, affine_theta=None, i=0):
        self.generator.train()

        N, C, H, W = original_image.size()

        input_image = original_image
        
        reconstructed, w, log_w, z, z_hat= self.generator(input_image, class_id)
       


        total_loss = 0

        rec_err = torch.abs(input_image - reconstructed)
        loss_rec = self.lambda_rec * torch.mean(rec_err)
        total_loss = total_loss + loss_rec


        if not self.no_ssim:
            ssim = SSIM(window_size = 11, size_average=True)
            loss_ssim = -self.lambda_ssim * ssim(input_image, reconstructed)
            total_loss = total_loss + loss_ssim
        

        if not self.no_memfeat_loss:
            loss_feat_compact, loss_feat_sep = self.generator.memory.update_memory_loss(z, w, class_id)  

            loss_feat_compact = self.lambda_feat_compact * loss_feat_compact   
            loss_feat_sep = self.lambda_feat_sep * loss_feat_sep   

            total_loss = total_loss + loss_feat_compact
            total_loss = total_loss + loss_feat_sep
       

        if not self.no_vgg_loss:
            if reconstructed.size(1) == 1:
                loss_vgg = self.lambda_vgg * self.vgg_loss(torch.cat([reconstructed]*3,dim=1), torch.cat([input_image]*3,dim=1))
            else:
                loss_vgg = self.lambda_vgg * self.vgg_loss(reconstructed,input_image)

            total_loss = total_loss + loss_vgg
        
        if not self.no_lpips_loss:

            loss_lpips = self.lambda_lpips * self.lpips_loss.forward(input_image.detach(),reconstructed).mean()
            total_loss = total_loss + loss_lpips

        if self.with_discriminator:
            if self.no_gan_feat_loss:
                fake_global_pred = self.discriminator(reconstructed)
                loss_gan_global = -torch.mean(fake_global_pred)

                total_loss = total_loss + loss_gan_global
            else:
                combined = torch.cat([input_image, reconstructed], dim=0)
                global_predictions, intermediate_features = self.discriminator(combined, output_features=True)
                real_global_pred = global_predictions[:N]
                fake_global_pred = global_predictions[N:]

                loss_gan_global = -torch.mean(fake_global_pred)

                loss_gan_features = 0
                for layer in intermediate_features:
                    real_features = layer[:N]
                    fake_features = layer[N:]
                    loss_gan_features = loss_gan_features + torch.mean(torch.abs(real_features.detach() - fake_features))
                loss_gan_features = self.lambda_gan_feat * loss_gan_features

                total_loss = total_loss + loss_gan_global + loss_gan_features

        self.g_optimizer.zero_grad()
        total_loss.backward()
        self.g_optimizer.step()

        loss = {}
        loss["train/G/loss_rec"] = loss_rec.data.item()
        if not self.no_ssim:
            loss["train/G/loss_ssim"] = loss_ssim.data.item()
        if not self.no_vgg_loss:
            loss["train/G/loss_vgg"] = loss_vgg.data.item()
        if not self.no_memfeat_loss:
            loss["train/G/loss_feat_compact"] = loss_feat_compact.data.item()
            loss["train/G/loss_feat_sep"] = loss_feat_sep.data.item()
        if self.with_discriminator:
            loss["train/G/loss_gan_global"] = loss_gan_global.data.item()
            if not self.no_gan_feat_loss:
                loss["train/G/loss_gan_features"] = loss_gan_features.data.item()
        if not self.no_lpips_loss:
            loss["train/G/loss_lpips"] = loss_lpips.data.item()
        loss["train/G/total_loss"] = total_loss.data.item()

        return loss
    
    def detect_anomaly(self, model, data_loader, mean, std, eval_type="rec", out_filename="training_error", confusion_mat=False):
        """[summary]

        Args:
            model ([torch sequentail model]): [description]
            data_loader ([dataloader]): [description]
            mean ([float]): [description]
            std ([float]): [description]
            eval_type (str, optional): [description]. Defaults to "rec". choices: [rec, ssim, mix]
        Return:
            detect_cnt([num]): The number of anomaly images
        """
        STD_TIMES_ERROR = 1
        print(f"Start detect_anomaly_cnt mean = {mean}, std = {std}, threshold = mean + {STD_TIMES_ERROR}*std")
        # assert eval_type in ["rec", "ssim", "mix"]
        detect_cnt = 0
        OK_cnt = 0
        NG_cnt = 0
        model.eval()
        data_iter = iter(data_loader)
        out_data = []
        y_true, y_pred = [], []
        for i, (input_image, class_id, im_id, affine_theta) in enumerate(tqdm(data_loader)):
            n_batch = input_image.size(0)
            # In this setting, we set dataloader batch size as 1.
            # assert n_batch == 1
            if torch.cuda.is_available():
                input_image = input_image.cuda()
                affine_theta = affine_theta.cuda()

            with torch.no_grad():
                reconstruction, w, log_w, z, z_hat = model(input_image, class_id)
                loss = 0
                # if eval_type in ["ssim", "mix"]:
                    # Shape = [B, 3, W, H]
                ssim_map = -ssim(input_image, reconstruction, size_average=False)
                # Shape = [B,]
                loss_ssim_map = ssim_map.view(n_batch, -1).mean(dim=1)
                loss += loss_ssim_map

                # if eval_type in ["rec", "mix"]:
                # Shape = [B, 3, W, H]
                loss_rec = torch.abs(input_image - reconstruction)
                # Shape = [B,]
                loss_rec = loss_rec.view(n_batch, -1).mean(dim=1)
                loss += loss_rec
                # Todo: Check std setting
                select_images = loss.abs() > mean + STD_TIMES_ERROR*std
                # if torch.any(select_images, 0):
                for idx, is_save in enumerate(select_images):
                    # is_save means detect error
                    is_save = str(is_save.data.item())
                    im_id_ = str(im_id[idx]).replace("/", "_").replace(".JPG", "")
                    y_true += ["OK" in im_id]   
                    y_pred += [not is_save] 
                    out_data += [[im_id_, float(loss[idx]), float(loss_rec[idx]), float(loss_ssim_map[idx]), is_save, out_filename]]
                    if not is_save:
                        continue
                    input_image_t = input_image[idx]
                    reconstruction_t = reconstruction[idx]
                    ssim_map_t = ssim_map[idx]
                    detect_cnt += 1
                    val_image = torch.cat([input_image_t, reconstruction_t, ssim_map_t], dim=2)
                    result_path = os.path.join(self.result_save_path, f'{detect_cnt}-{im_id_}-{out_filename}.jpg')
                    print(f"Detect! detect_cnt = {detect_cnt}, save as {result_path}")
                    save_image((val_image + 1) / 2, result_path, nrow=2, padding=1)

        # Save to csv
        df = pd.DataFrame(out_data, columns=["filename", "total_loss", "loss_rec", "loss_ssim_map", "detect", "dataset_type"])
        df.to_csv(out_filename)
        # Save to confusion_matrix
        if confusion_mat:
            print(f"y_true = {len(y_true)}")
            print(f"y_pred = {len(y_pred)}")
            (tn, fp, fn, tp) = confusion_matrix(y_true, y_pred).ravel()
            print(f"tp = {tp} fp = {fp} tn = {tn} fn = {fn}")



        return int(detect_cnt)
        
        
    def get_std_by_validation(self, model, data_loader, prefix=""):
        model.eval()
        data_iter = iter(data_loader)
        n_counter = 0
        loss = {
                "val/ssim": [],
                "val/rec":[],
                "val/total":[]
            }
        
        for i, (input_image, class_id, im_id, affine_theta) in enumerate(tqdm(data_loader)):
            n_batch = input_image.size(0)
            if torch.cuda.is_available():
                input_image = input_image.cuda()
                affine_theta = affine_theta.cuda()

            with torch.no_grad():
                reconstruction, w, log_w, z, z_hat = model(input_image, class_id)
                # TODO: Consider remove ssim
                # Shape = [B, 3, W, H]
                ssim_map = -ssim(input_image, reconstruction, size_average=False)
                # Shape = [B, 3, W, H]
                loss_rec = torch.abs(input_image - reconstruction)
                # loss_rec = torch.abs(input_image - reconstruction)
                # Shape [B, 3*W*H]
                # TODO:Image-level loss, instead of pix-level loss
                ssim_map = ssim_map.view(n_batch, -1).mean(dim=1)
                loss_rec = loss_rec.view(n_batch, -1).mean(dim=1)
                total_loss = ssim_map + loss_rec
                # if not self.grayscale:
                    # ssim_map = ssim_map.mean(dim=1,keepdim=True)
                    # ssim_map = torch.cat([ssim_map]*3,dim=1)


                # val_image = torch.cat([input_image, reconstruction, ssim_map], dim=3)
                # result_path = os.path.join(self.result_save_path, '{}-{}-val.jpg'.format(train_iters, i))
                # save_image((val_image + 1) / 2, result_path, nrow=2, padding=1)

            loss["val/ssim"] += [single_ssim_map.mean().data.item() for single_ssim_map in ssim_map]
            loss["val/rec"] += [single_loss_rec.data.item() for single_loss_rec in loss_rec]
            loss["val/total"] += [single_total_loss.data.item() for single_total_loss in total_loss]
            
            
            n_counter += 1
            
        loss["ssim_mean"], loss["ssim_std"] = torch.std_mean(torch.FloatTensor(loss["val/ssim"]), dim=0)
        loss["rec_mean"], loss["rec_std"] = torch.std_mean(torch.FloatTensor(loss["val/rec"]), dim=0)
        loss["total_mean"], loss["total_std"] = torch.std_mean(torch.FloatTensor(loss["val/total"]), dim=0)
        loss["val/ssim"] = sum(loss["val/ssim"]) / len(loss["val/ssim"])
        loss["val/rec"] = sum(loss["val/rec"]) / len(loss["val/rec"])
        loss["val/total"] = sum(loss["val/total"]) / len(loss["val/total"])
        # print(loss["val/ssim"])
        return loss
        
    def get_test_perceptual_loss(self, model, data_loader):
        model.eval()
        data_iter = iter(data_loader)
        n_counter = 0
        lpips_loss_list = []

        for i, (input_image, class_id, im_id, affine_theta) in enumerate(tqdm(data_loader)):
            n_batch = input_image.size(0)
            if torch.cuda.is_available():
                input_image = input_image.cuda()
                affine_theta = affine_theta.cuda()

            with torch.no_grad():
                reconstruction, w, log_w, z, z_hat = model(input_image, class_id)

                loss_lpips_map = self.lambda_lpips * self.lpips_loss.forward(input_image.detach(),reconstruction)
                loss_lpips_value = loss_lpips_map.mean()

            lpips_loss_list.append(loss_lpips_value)

        # print(loss["val/ssim"])
        return lpips_loss_list
        
    def sample_validation(self, model, data_loader, prefix="",train_iters=None, val_num_iters=None):
        if self.norm_type == "batch" or self.norm_type == "cbatch" or self.norm_type == "ginstance":
            model.eval()
        n_counter = 0
        data_iter = iter(data_loader)
        avg_loss = {}
        if val_num_iters is None:
            val_num_iters = len(data_loader)

        image_val_list = {}
        orig_val_list = {}
        rec_val_list = {}
        predictions = []
        for i, (input_image, class_id, im_id, affine_theta) in enumerate(tqdm(data_loader)):

            if torch.cuda.is_available():
                input_image = input_image.cuda()
                affine_theta = affine_theta.cuda()

            with torch.no_grad():
                ssim = SSIM(window_size = 11, size_average=False)
                reconstruction, w, log_w, z, z_hat = model(input_image, class_id)
                    
                ssim_map = -ssim(input_image, reconstruction)
                loss_rec = 10*torch.mean(torch.abs(input_image - reconstruction))
                total_loss = ssim_map.mean() + loss_rec
                if not self.grayscale:
                    ssim_map = ssim_map.mean(dim=1,keepdim=True)
                    ssim_map = torch.cat([ssim_map]*3,dim=1)


                val_image = torch.cat([input_image, reconstruction, ssim_map], dim=3)

                result_path = os.path.join(self.result_save_path, '{}-{}-val.jpg'.format(train_iters, i))

                save_image((val_image + 1) / 2, result_path, nrow=2, padding=1)

            loss = {}
            loss["val/ssim"] = ssim_map.mean().data.item()
            loss["val/rec"] = loss_rec.data.item()
            loss["val/total"] = total_loss.data.item()

            n_batch = input_image.size(0)
            n_total = n_counter + n_batch
            
            for key in loss:
                old_mu = avg_loss.setdefault(key, 0)
                mu_batch = loss[key]

                new_mu = mu_batch - (n_counter / n_total) * mu_batch + old_mu - (n_batch / n_total) * old_mu
                avg_loss[key] = new_mu

            n_counter += n_batch

        
        

        
        del ssim_map
        del input_image
        del val_image
        del reconstruction
        torch.cuda.empty_cache()

        return avg_loss

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)
