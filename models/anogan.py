
import numpy as np
from tqdm import tqdm
import json
import cv2
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch.backends.cudnn as cudnn


from lib.utils import *
from lib.evaluate import evaluate
from lib.train_gan import GANBaseModel

def predict_forg(gout, input):
    # predict image
    # diff video
    diff = torch.abs(gout - input)
    # normalize diff -> (0, 1)
    diff_norm = [normalize(v) for v in diff.permute(2, 0, 1, 3, 4)]
    diff_norm = torch.stack(diff_norm).permute(1, 0, 3, 4, 2) # (B, D, W, H, C)
    # tensor to numpy
    diff_norm = diff_norm.cpu().numpy()
    # post processing 
    diff_norm = rgb_to_gray(diff_norm)
    predict = diff_norm
    #predict = np.expand_dims(morphology_proc(predict), axis=1)
    return torch.from_numpy(predict).permute(0, 4, 1, 2, 3).to('cuda')

class NetG(nn.Module):
    def __init__(self):
        super(NetG, self).__init__()
        
        self.layer1 = nn.Sequential(
                    nn.Linear(100, 2*512*16*16),
                    nn.BatchNorm1d(2*512*16*16),
                    nn.ReLU(),
                    )
        # 512x2x16x16
        self.layer2 = nn.Sequential(
                    nn.Dropout(p=0.25),
                    nn.ConvTranspose3d(512, 256, 3, 2, 1, 1), # 256x4x32x32
                    nn.Conv3d(256, 256, 3, 1, 1), # 256x4x32x32
                    nn.BatchNorm3d(256),
                    nn.LeakyReLU(),
                    nn.Dropout(p=0.25),
                    nn.ConvTranspose3d(256, 128, 3, 2, 1, 1), # 128x8x64x64
                    nn.Conv3d(128, 128, 3, 1, 1), # 256x4x32x32
                    nn.BatchNorm3d(128),
                    nn.LeakyReLU()
                    )

        self.layer3 = nn.Sequential(
                    nn.Dropout(p=0.25),
                    nn.ConvTranspose3d(128, 64, 3, 1, 1), # 64x8x64x64
                    nn.Conv3d(64, 64, 3, 1, 1), # 256x4x32x32
                    nn.BatchNorm3d(64),
                    nn.LeakyReLU(),
                    nn.Dropout(p=0.25),
                    nn.ConvTranspose3d(64, 3, 3, 2, 1, 1), # 3x16x128x128
                    nn.Conv3d(3, 3, 3, 1, 1), # 256x4x32x32
                    nn.Sigmoid()
                    )

    def forward(self, z):
        x = self.layer1(z)
        x = x.view(x.size()[0], 512, 2, 16, 16)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class NetD(nn.Module):
    def __init__(self):
        super(NetD, self).__init__()
        self.layer1 = nn.Sequential(
                    nn.Conv3d(3, 32, 3, stride=1, padding=1), # 32x16x128x128
                    nn.BatchNorm3d(32),
                    nn.LeakyReLU(),
                    nn.Conv3d(32, 64, 3, stride=1, padding=1), # 64x16x128x128
                    nn.Conv3d(64, 64, 3, stride=1, padding=1), # 64x16x128x128
                    nn.BatchNorm3d(64),
                    nn.LeakyReLU(64),
                    nn.AvgPool3d(2) #64x8x64x64
                )

        self.layer2 = nn.Sequential(
                    nn.Conv3d(64, 128, 3, stride=1, padding=1), # 128x8x64x64
                    nn.Conv3d(128, 128, 3, stride=1, padding=1), # 128x8x64x64
                    nn.BatchNorm3d(128),
                    nn.LeakyReLU(),
                    nn.AvgPool3d(2), # 128x4x32x32
                    nn.Conv3d(128, 256, 3, stride=1, padding=1), # 256x4x32x32
                    nn.BatchNorm3d(256),
                    nn.LeakyReLU(),
                    nn.AvgPool3d(2) # 256x2x16x16
                )

        self.fc = nn.Sequential(
                    nn.Linear(256*2*16*16, 1),
                    nn.Sigmoid()
                )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size()[0], -1)
        feature = x
        out = self.fc(x)

        return out, feature

class AnoGAN(GANBaseModel):
    def __init__(self, args, dataloader):
        super(AnoGAN, self).__init__(args, dataloader)

        # Create and initialize networkgs.
        if len(self.args.gpu) > 1:
            self.netg = torch.nn.DataParallel(NetG(), device_ids=self.args.gpu, dim=0)
            self.netd = torch.nn.DataParallel(NetD(), device_ids=self.args.gpu, dim=0)
            self.netg = self.netg.cuda()
            self.netd = self.netd.cuda()
            cudnn.benchmark = True
        else:
            self.netg = NetG().to('cuda')
            self.netd = NetD().to('cuda')
        self.netg.apply(weights_init)
        self.netd.apply(weights_init)

        self.loss = nn.BCELoss()
        self.g_opt = torch.optim.Adam(self.netg.parameters(), lr=5*args.lr, betas=(0.5, 0.999))
        self.d_opt = torch.optim.Adam(self.netd.parameters(), lr=args.lr, betas=(0.5, 0.999))

        self.ones_label = torch.ones(args.batchsize).cuda()
        self.zeros_label = torch.zeros(args.batchsize).cuda()
   
    def test(self):
        self.netg.eval()
        self.netd.eval()

        gen_loss_ = []
        dis_loss_real_ = []
        dis_loss_fake_ = []
        dis_loss_ = []

        gts = []
        predicts = []

        with torch.no_grad():
            # Test
            pbar = tqdm(self.dataloader['test'], leave=True, ncols=100, total=len(self.dataloader['test']))
            for i, data in enumerate(pbar):
                
                # set test data 
                input, real, gt, lb = (d.to('cuda') for d in data)
                 
                # NetD

                dis_real_ = self.netd(real)[0].view(-1)
                dis_loss_real_.append(self.loss(dis_real_, self.ones_label).item())

                z = torch.randn(self.args.batchsize, 100, device='cuda')
                gen_fake_ = self.netg(z)
                dis_fake_ = self.netd(gen_fake_.detach())[0].view(-1)
                dis_loss_fake_.append(self.loss(dis_fake_, self.zeros_label).item())
                dis_loss_.append(dis_loss_real_[-1] + dis_loss_fake_[-1])

                # NetG
                dis_fake_ = self.netd(gen_fake_)[0].view(-1)
                gen_loss_.append(self.loss(dis_fake_, self.ones_label).item())
                
                predict_ = predict_forg(gen_fake_, real)
                t_pre_ = threshold(predict_.detach())
                m_pre_ = morphology_proc(t_pre_)

                gts.append(gt.permute(0, 2, 3, 4, 1).cpu().numpy())
                predicts.append(predict_.permute(0, 2, 3, 4, 1).cpu().numpy())
                
                # test video summary
                self.color_video_dict.update({
                        'test/input-real-gen': torch.cat([input, real, gen_fake_], dim=3),
                    })
                self.gray_video_dict.update({
                        'test/gt-pre-th-morph': torch.cat([gt, predict_, t_pre_, m_pre_], dim=3)
                    })
                self.hist_dict.update({
                    "test/inp": input,
                    "test/gt": gt,
                    "test/gen": gen_fake_,
                    "test/predict": predict_,
                    "test/t_pre": t_pre_,
                    "test/m_pre": m_pre_
                    })

                pbar.set_description("[TEST  Epoch %d/%d]" % (self.epoch+1, self.args.ep))

            # AUC
            gts = np.asarray(np.stack(gts), dtype=np.int32).flatten()
            predicts = np.asarray(np.stack(predicts)).flatten()
            roc = evaluate(gts, predicts, self.best_roc, self.epoch, self.save_root_dir, metric='roc')
            pr = evaluate(gts, predicts, self.best_pr, self.epoch, self.save_root_dir, metric='pr')
            f1 = evaluate(gts, predicts, metric='f1_score')
            if roc > self.best_roc: 
                self.best_roc = roc
                self.save_weights('roc')
            elif pr > self.best_pr:
                self.best_pr = pr
                self.save_weights('pr')
                
            # Update summary of loss ans auc
            self.score_dict.update({
                    "score/roc": roc,
                    "score/pr": pr,
                    "score/f1": f1
                })
            self.errors_dict.update({
                        'd/err_d/test': np.mean(gen_loss_),
                        'd/err_g/test': np.mean(dis_loss_)
                        })

    def optimize_params(self):
        # NetD
        self.netd.zero_grad()
        
        dis_real = self.netd(self.real)[0].view(-1)
        dis_loss_real = self.loss(dis_real, self.ones_label)
        dis_loss_real.backward()
        
        z = torch.randn(self.args.batchsize, 100, device='cuda')
        gen_fake = self.netg(z)
        dis_fake = self.netd(gen_fake.detach())[0].view(-1)
        dis_loss_fake = self.loss(dis_fake, self.zeros_label)
        dis_loss_fake.backward()
        dis_loss = dis_loss_real + dis_loss_fake
        self.d_opt.step()

        # NetG
        self.netg.zero_grad()
        dis_fake = self.netd(gen_fake)[0].view(-1)
        gen_loss = self.loss(dis_fake, self.ones_label)
        gen_loss.backward(retain_graph=True)
        self.g_opt.step()
        
        predict = predict_forg(gen_fake.detach(), self.real)
        t_pre = threshold(predict.detach())
        m_pre = morphology_proc(t_pre)
        
        self.color_video_dict.update({
            "train/input-real-gen": torch.cat([self.input, self.real, gen_fake], dim=3),
            })
        self.gray_video_dict.update({
            "train/gt-pre-th-mor": torch.cat([self.gt, predict, t_pre, m_pre], dim=3)
            })
        
        self.errors_dict.update({
            'd/err_d/train': dis_loss.item(),
            'g/err_g/train': gen_loss.item()
            })
    

                

