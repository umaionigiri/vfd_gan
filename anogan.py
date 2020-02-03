
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


from utils import *
from evaluate import evaluate

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

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.layer1 = nn.Sequential(
                    nn.Linear(100, 2*512*16*16),
                    nn.BatchNorm1d(2*512*16*16),
                    nn.ReLU(),
                    )
        # 512x2x16x16
        self.layer2 = nn.Sequential(
                    nn.ConvTranspose3d(512, 256, 3, 2, 1, 1), # 256x4x32x32
                    nn.BatchNorm3d(256),
                    nn.LeakyReLU(),
                    nn.ConvTranspose3d(256, 128, 3, 2, 1, 1), # 128x8x64x64
                    nn.BatchNorm3d(128),
                    nn.LeakyReLU(),
                    )

        self.layer3 = nn.Sequential(
                    nn.ConvTranspose3d(128, 64, 3, 1, 1), # 64x8x64x64
                    nn.BatchNorm3d(64),
                    nn.LeakyReLU(),
                    nn.ConvTranspose3d(64, 3, 3, 2, 1, 1), # 3x16x128x128
                    nn.Tanh()
                    )

    def forward(self, z):
        x = self.layer1(z)
        x = x.view(x.size()[0], 512, 2, 16, 16)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
                    nn.Conv3d(3, 8, 3, stride=1, padding=1), # 8x16x128x128
                    nn.BatchNorm3d(8),
                    nn.LeakyReLU(),
                    nn.Conv3d(8, 16, 3, stride=2, padding=1), # 16x8x64x64
                    nn.BatchNorm3d(16),
                    nn.LeakyReLU(16),
                )

        self.layer2 = nn.Sequential(
                    nn.Conv3d(16, 32, 3, stride=2, padding=1), # 32x4x32x32
                    nn.BatchNorm3d(32),
                    nn.LeakyReLU(),
                    nn.Conv3d(32, 64, 3, stride=2, padding=1), # 64x2x16x16
                    nn.BatchNorm3d(64),
                    nn.LeakyReLU(),
                )

        self.fc = nn.Sequential(
                    nn.Linear(64*2*16*16, 1),
                    nn.Sigmoid()
                )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size()[0], -1)
        feature = x
        out = self.fc(x)

        return out, feature

class AnoGAN():
    def __init__(self, args, dataloader):
        super(AnoGAN, self).__init__()
        # Create and initialize networkgs.

        self.args = args
        self.dataloader = dataloader

        self.global_step = 0
        self.best_roc = 0
        self.best_pr = 0
        self.color_video_dict = OrderedDict()
        self.gray_video_dict = OrderedDict()
        self.train_errors_dict = {}
        self.test_errors_dict = {}
        self.hist_dict = OrderedDict()
        self.score_dict = OrderedDict()
 
        # make save root dir
        from datetime import datetime
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        comment = "b{}xd{}xwh{}_lr-{}".format(args.batchsize, args.nfr, args.isize, args.lr)
        self.save_root_dir = os.path.join(args.result_root, args.model, comment, current_time)
        if not os.path.exists(self.save_root_dir): os.makedirs(self.save_root_dir)
        # make weights save dir
        self.weight_dir = os.path.join(self.save_root_dir,'weights')
        if not os.path.exists(self.weight_dir): os.makedirs(self.weight_dir)
        # make tensorboard logdir 
        logdir = os.path.join(self.save_root_dir, "runs")
        if not os.path.exists(logdir): os.makedirs(logdir)
        self.writer = SummaryWriter(log_dir=logdir)
        #save args
        with open(self.save_root_dir+"/args.txt", mode="w") as f:
            json.dump(args.__dict__, f, indent=4)

        print("\n SAVE PATH == {} \n".format(self.save_root_dir))


        if len(self.args.gpu) > 1:
            self.netg = torch.nn.DataParallel(Generator(), device_ids=self.args.gpu, dim=0)
            self.netd = torch.nn.DataParallel(Discriminator(), device_ids=self.args.gpu, dim=0)
            self.netg = self.netg.cuda()
            self.netd = self.netd.cuda()
            cudnn.benchmark = True
        else:
            self.netg = Generator().to('cuda')
            self.netd = Discriminator().to('cuda')
        self.netg.apply(weights_init)
        self.netd.apply(weights_init)

        self.loss = nn.MSELoss()
        self.g_opt = torch.optim.Adam(self.netg.parameters(), lr=5*args.lr, betas=(0.5, 0.999))
        self.d_opt = torch.optim.Adam(self.netd.parameters(), lr=args.lr, betas=(0.5, 0.999))

        self.ones_label = Variable(torch.ones(args.batchsize, 1)).cuda()
        self.zeros_label = Variable(torch.zeros(args.batchsize, 1)).cuda()

    def update_summary(self):
        # VIDEO
        for t, v in self.color_video_dict.items():
            grid = [make_grid(f, nrow=self.args.batchsize, normalize=True) for f in v.permute(2, 0, 1, 3, 4)]
            self.writer.add_video(t, torch.unsqueeze(torch.stack(grid), 0), self.global_step)

        for t, v in self.gray_video_dict.items():
            grid = [make_grid(f, nrow=self.args.batchsize, normalize=False) for f in v.permute(2, 0, 1, 3, 4)]
            self.writer.add_video(t, torch.unsqueeze(torch.stack(grid), 0), self.global_step)

        # ERROR
        for t, e in self.train_errors_dict.items():
            self.writer.add_scalar(t, e, self.global_step)
        for t, e in self.test_errors_dict.items():
            self.writer.add_scalar(t, e, self.global_step)

        # HISTOGRAM
        for t, h in self.hist_dict.items():
            self.writer.add_histogram(t, h)

        # SCORE
        for t, s in self.score_dict.items():
            self.writer.add_scalar(t, s, self.global_step)

    def save_weights(self, name_head):
        # -- Save Weights of NetG and NetD --
        torch.save({'epoch': self.epoch + 1, 'state_dict': self.netg.state_dict()},
                   '%s/%s_ep%04d_netG.pth' % (self.weight_dir, name_head, self.epoch))
        torch.save({'epoch': self.epoch + 1, 'state_dict': self.netd.state_dict()},
                   '%s/%s_ep%04d_netD.pth' % (self.weight_dir, name_head, self.epoch))
    
    def test(self):
        self.netg.eval()
        self.netd.eval()

        gen_loss_ = []
        dis_loss_ = []

        gts = []
        predicts = []

        with torch.no_grad():
            # load weights 
            if self.args.load_weights != " " :
                path = self.args.load_weights
                pretrained_dict = torch.load(path)['state_dict']

                try:
                    self.netg.load_state_dict(pretrained_dict)
                except IOError:
                    raise IOError("netG weights not found")
                print("Loaded weights.")

            # Test
            pbar = tqdm(self.dataloader['test'], leave=True, ncols=100, total=len(self.dataloader['test']))
            for i, data in enumerate(pbar):
                
                # set test data 
                input, real, gt, lb = (d.to('cuda') for d in data)

                # NetG
                z = Variable(init.normal_(torch.Tensor(self.args.batchsize, 100), mean=0, std=0.1)).cuda()
                gen_fake_ = self.netg.forward(z)
                dis_fake_, _ = self.netd.forward(gen_fake_)
                
                gen_loss_.append(self.loss(dis_fake_, self.ones_label).item())
                
                # NetD
                #z = Variable(init.normal_(torch.Tensor(self.args.batchsize, 100), mean=0, std=0.1)).cuda()
                #gen_fake_ = self.netg.forward(z)
                #dis_fake_, _ = self.netd.forward(gen_fake_)
                dis_real_, _ = self.netd.forward(real)
                dis_loss_.append(self.loss(dis_fake_, self.zeros_label).item() + \
                                self.loss(dis_real_, self.ones_label).item())
                
                predict = predict_forg(gen_fake_, real)
                t_pre_ = threshold(predict.detach())
                m_pre_ = morphology_proc(t_pre_)

                gts.append(gt.permute(0, 2, 3, 4, 1).cpu().numpy())
                predicts.append(predict.permute(0, 2, 3, 4, 1).cpu().numpy())
                
                # test video summary
                self.color_video_dict.update({
                        'test/input-real-gen': torch.cat([input, real, gen_fake_], dim=3),
                    })
                self.gray_video_dict.update({
                        'test/gt-pre-th-morph': torch.cat([gt, predict, t_pre_, m_pre_], dim=3)
                    })
                self.hist_dict.update({
                    "test/inp": input,
                    "test/gt": gt,
                    "test/gen": gen_fake_,
                    "test/predict": predict,
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
            self.test_errors_dict.update({
                        'test/err_d': np.mean(gen_loss_),
                        'test/err_g': np.mean(dis_loss_)
                        })

    def train(self):
       
        for self.epoch in range(self.args.ep):

            pbar  = tqdm(self.dataloader['train'], leave=True, ncols=100,
                                    total=len(self.dataloader['train']))

            for i, data in enumerate(pbar):
                    
                self.global_step += 1

                inp, real, gt, lb = (d.to('cuda') for d in data)
                # Generator
                self.netg.train()
                self.g_opt.zero_grad()
                z = Variable(init.normal_(torch.Tensor(self.args.batchsize, 100), mean=0, std=0.1)).cuda()
                gen_fake = self.netg(z)
                dis_fake, _ = self.netd(gen_fake)
                gen_loss = torch.sum(self.loss(dis_fake, self.ones_label))
                gen_loss.backward(retain_graph=True)
                self.g_opt.step()
                
                # Discriminator
                self.netd.train()
                self.d_opt.zero_grad()
                gen_fake = self.netg(z)
                dis_fake, _ = self.netd.forward(gen_fake)
                dis_real, _ = self.netd.forward(real)
                dis_loss = torch.sum(self.loss(dis_fake, self.zeros_label)) + \
                            torch.sum(self.loss(dis_real, self.ones_label))
                dis_loss.backward()
                self.d_opt.step()

                predict = predict_forg(gen_fake.detach(), real)
                t_pre = threshold(predict.detach())
                m_pre = morphology_proc(t_pre)

                self.color_video_dict.update({
                            "train/input-real-gen": torch.cat([inp, real, gen_fake], dim=3),
                        })
                self.gray_video_dict.update({
                            "train/gt-pre-th-mor": torch.cat([gt, predict, t_pre, m_pre], dim=3)
                        })
 
                self.train_errors_dict.update({
                        'train/err_d': dis_loss.item(),
                        'train/err_g': gen_loss.item()
                    })

                if self.global_step % self.args.test_freq == 0:
                    self.test()
                if self.global_step % self.args.display_freq == 0:
                    self.update_summary()
                    
                pbar.set_description("[TRAIN Epoch %d/%d]" % (self.epoch+1, self.args.ep))

        print("Training model Done.")
 









