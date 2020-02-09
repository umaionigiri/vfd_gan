
import os
import numpy as np
from tqdm import tqdm
import json
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch.backends.cudnn as cudnn

from lib.utils import *


class GANBaseModel():
    def __init__(self, args, dataloader):

        self.args = args
        self.dataloader = dataloader

        self.global_step = 0
        self.best_roc = 0
        self.best_pr = 0
        self.color_video_dict = OrderedDict()
        self.gray_video_dict = OrderedDict()
        self.errors_dict = {}
        self.hist_dict = OrderedDict()
        self.score_dict = OrderedDict()
        
        # make save root dir
        from datetime import datetime
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        comment = "b{}xd{}xwh{}_lr-{}_w-a{}c{}".format(args.batchsize, args.nfr, args.isize, 
                                                    args.lr, args.w_adv, args.w_con)
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

    def save_weights(self, name_head):
        # -- Save Weights of NetG and NetD --
        torch.save({'epoch': self.epoch + 1, 'state_dict': self.netg.state_dict()},
                   '%s/%s_ep%04d_netG.pth' % (self.weight_dir, name_head, self.epoch))
        torch.save({'epoch': self.epoch + 1, 'state_dict': self.netd.state_dict()},
                   '%s/%s_ep%04d_netD.pth' % (self.weight_dir, name_head, self.epoch))

    def train(self):

        print(" >> Training model %s." % self.args.model)

        for self.epoch in range(self.args.ep):
            pbar = tqdm(self.dataloader['train'], leave=True, ncols=100, total=len(self.dataloader['train']))
            for i, data in enumerate(pbar):

                self.global_step += 1

                self.input, self.real, self.gt, self.lb = (d.to('cuda') for d in data)
                self.optimize_params()

                if self.global_step % self.args.freq == 0:
                    # -- TEST --
                    self.test()
                
                if self.global_step % self.args.freq == 0:
                    # -- Update Summary on Tensorboard --
                    update_summary(self.writer, self.args.batchsize, self.global_step, 
                                    self.color_video_dict, self.gray_video_dict,
                                    self.errors_dict, self.score_dict)

                pbar.set_description("[TRAIN Epoch %d/%d]" % (self.epoch+1, self.args.ep))
     
            
        print(" >> Training model %s.[Done]" % self.args.model)


