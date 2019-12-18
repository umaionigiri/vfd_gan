
import os
from tqdm import tqdm
import numpy as np
import json
import cv2

from collections import OrderedDict
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch.backends.cudnn as cudnn

from evaluate import evaluate
from networks_new import NetG, NetD, weights_init
from utils import *


class BaseModel():
    """
    Base Model for ganomaly

    """

    def __init__(self, args, dataloader):

        self.args = args
        self.dataloader = dataloader

        self.train_iter = 0
        self.test_iter = 0
        self.train_imgs_dict = OrderedDict()
        self.test_imgs_dict = OrderedDict()
        self.train_errors_dict = OrderedDict()
        self.test_errors_dict = OrderedDict()
        self.auc_dict = OrderedDict()
        
        # set using gpu
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # make save root dir
        from datetime import datetime
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        comment = "b{}xd{}xwh{}_lr-{}_w-a{}c{}p{}".format(args.batchsize, args.nfr, args.isize, 
                                                    args.lr, args.w_adv, args.w_con, args.w_pre)
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

    def set_input(self, input:torch.Tensor):
        with torch.no_grad():
            self.input.resize_(input[0].size()).copy_(input[0])
            self.gt.resize_(input[1].size()).copy_(input[1])
            self.label.resize_(input[2].size())

    def train_one_epoch(self):

        self.netg.train()
        self.netd.train()

        pbar = tqdm(self.dataloader['train'], leave=True, ncols=100, total=len(self.dataloader['train']))
        for i, data in enumerate(pbar):
            self.train_iter += 1

            self.set_input(data)
            self.optimize_params()
            
            if self.train_iter % self.args.display_freq == 0:
                # -- Update Summary on Tensorboard --
                reals = self.input.data
                gout = self.gout
                real_flow = self.input_flow
                fake_flow = self.gout_flow

                self.train_errors_dict.update({
                                ('train/err_d', self.err_d.item()),
                                ('train/err_g', self.err_g.item()),
                                ('train/err_g_adv', self.err_g_adv.item()),
                                ('train/err_g_adv_s', self.err_g_adv_s.item()),
                                ('train/err_g_adv_t', self.err_g_adv_t.item()),
                                ('train/err_g_con', self.err_g_con.item()),
                                ('train/err_g_pre', self.err_g_pre.item()),
                                })
                self.train_imgs_dict.update({
                        "train/input,gen,input_flow, gen_flow": torch.cat([reals, gout, real_flow, fake_flow], dim=3),
                    })
                for tag, err in self.train_errors_dict.items():
                    self.writer.add_scalar(tag, err, self.train_iter)
                for tag, v in self.train_imgs_dict.items():
                    grid = [make_grid(f, nrow=self.args.batchsize, normalize=True) 
                            for f in v.permute(2, 0, 1, 3, 4)]
                    self.writer.add_video(tag, torch.unsqueeze(torch.stack(grid), 0), self.train_iter)

            pbar.set_postfix(OrderedDict(loss="{:.4f}".format(self.err_g)))
            pbar.set_description("[TRAIN Epoch %d/%d]" % (self.epoch+1, self.args.ep))

       
    def test_epoch_end(self):
        
        self.netg.eval()
        self.netd.eval()

        err_g_adv_s = []
        err_g_adv_t = []
        err_g_adv = []
        err_g_con = []
        err_g_pre = []
        err_g = []
        err_d = []

        predicts = []
        gts = []

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
                self.test_iter += 1
                # set test data 
                self.set_input(data) # get self.input, self.gt
                # NetG
                gout = self.netg(self.input) # Reconstract self.input
                predict = predict_forg(self.input, gout)
                gt = self.gt.permute(0, 2, 3, 4, 1).cpu().numpy()
                predicts.append(predict)
                gts.append(gt)
                # NetD
                # calc Optical Flow 
                input_flow = video_to_flow(self.input.detach()).to(self.device)
                gout_flow = video_to_flow(gout.detach()).to(self.device)
                # get disc output
                s_pred_real, s_feat_real, t_pred_real, t_feat_real \
                                        = self.netd(self.input, input_flow.detach())
                s_pred_fake, s_feat_fake, t_pred_fake, t_feat_fake \
                                        = self.netd(gout.detach(), gout_flow.detach())
                # Calc err_g
                err_g_adv_s.append(self.l_adv(s_feat_real, s_feat_fake).item())
                err_g_adv_t.append(self.l_adv(t_feat_real, t_feat_fake).item())
                err_g_adv.append(err_g_adv_s[-1] + err_g_adv_t[-1])
                err_g_con.append(self.l_con(gout, self.input).item())
                err_g_pre.append(self.l_bce(torch.from_numpy(predict).to(self.device), self.gt).item())
                err_g.append(err_g_adv[-1] * self.args.w_adv + err_g_con[-1] * self.args.w_con + err_g_pre[-1] * self.args.w_pre)
                # Calc err_d
                err_d_real_s = self.l_bce(s_pred_real, self.real_label).item()
                err_d_real_t = self.l_bce(t_pred_real, self.real_label).item()
                err_d_fake_s = self.l_bce(s_pred_fake, self.gout_label).item()
                err_d_fake_t = self.l_bce(t_pred_fake, self.gout_label).item()
                err_d_real = (err_d_real_s + err_d_real_t) * 0.5
                err_d_fake = (err_d_fake_s + err_d_fake_t) * 0.5
                err_d.append((err_d_real + err_d_fake) * 0.5)
                
                # test video summary
                self.test_imgs_dict.update({
                        'test/input_gout': torch.cat([self.input, gout], dim=3),
                        'test/gt_predict': torch.cat([self.gt, torch.from_numpy(predict).to(self.device)], dim=3)
                    })
                for t, v in self.test_imgs_dict.items():
                    grid = [make_grid(f, nrow=self.args.batchsize, normalize=True) for f in v.permute(2, 0, 1, 3, 4)]
                    self.writer.add_video(t, torch.unsqueeze(torch.stack(grid), 0), self.test_iter)
 
                pbar.set_description("[TEST  Epoch %d/%d]" % (self.epoch+1, self.args.ep))

            # AUC
            gts = np.asarray(np.stack(gts), dtype=np.int32).flatten()
            predicts = np.asarray(np.stack(predicts)).flatten()
            auc = evaluate(gts, predicts, self.save_root_dir, metric=self.args.metric)

            # Update summary of loss ans auc
            self.writer.add_scalar('auc', auc, self.epoch)
            self.test_errors_dict.update({
                        ('test/err_d', np.mean(err_d)),
                        ('test/err_g', np.mean(err_g)),
                        ('test/err_g_adv', np.mean(err_g_adv)),
                        ('test/err_g_adv_s', np.mean(err_g_adv_s)),
                        ('test/err_g_adv_t', np.mean(err_g_adv_t)),
                        ('test/err_g_con', np.mean(err_g_con)),
                        ('test/err_g_pre', np.mean(err_g_pre)),
                        })

            for tag, err in self.test_errors_dict.items():
                self.writer.add_scalar(tag, err, self.epoch)
 

    def train(self):

        best_auc = 0
        phase = self.args.phase
        print(" >> Training model %s." % self.args.model)

        for self.epoch in range(self.args.ep):
            if phase == 'train':
                self.train_one_epoch()
                phase = 'test'
            if phase == 'test':
                self.test_epoch_end()
                if self.args.phase == 'train': phase = 'train'

            if self.epoch % self.args.save_weight_freq == 0:
                torch.save({'epoch': self.epoch + 1, 'state_dict': self.netg.state_dict()},
                           '%s/ep%04d_netG.pth' % (self.weight_dir, self.epoch))
                torch.save({'epoch': self.epoch + 1, 'state_dict': self.netd.state_dict()},
                           '%s/ep%04d_netD.pth' % (self.weight_dir, self.epoch))

            print(">> Training model %s. Epoch %d/%d " % (self.name, self.epoch+1, self.args.ep))
            
        print(" >> Training model %s.[Done]" % self.args.model)



class Ganomaly(BaseModel):

    @property
    def name(self): return 'Ganomaly'

    def __init__(self, args, dataloader):
        super(Ganomaly, self).__init__(args, dataloader)

        # -- Misc attributes
        self.epoch = 0

        inp_shape = (self.args.batchsize, self.args.ich, 
                    self.args.nfr, self.args.isize, self.args.isize)

        # Create and initialize networkgs.
        if len(self.args.gpu) > 1:
            self.netg = torch.nn.DataParallel(NetG(), device_ids=self.args.gpu, dim=0)
            self.netd = torch.nn.DataParallel(NetD(self.args), device_ids=self.args.gpu, dim=0)
            self.netg = self.netg.cuda()
            self.netd = self.netd.cuda()
            cudnn.benchmark = True
        else:
            self.netg = NetG().to(self.device)
            self.netd = NetD(self.args).to(self.device)
        self.netg.apply(weights_init)
        self.netd.apply(weights_init)

        #pre-trained network load
        if self.args.resume != '' :
            print("\n Loading pre-trained networks.")
            self.args.iter = torch.load(os.path.join(self.resume, 'netG.pth'))['epoch']
            self.netg.load_state_dict(torch.load(os.path.join(self.args.resume, \
                                                            'netG.pth'))['state_dict'])
            self.netd.load_state_dict(torch.load(os.path.join(self.args.resume, 
                                                            'netD.pth'))['state_dict'])
            print("\t Done. \n")


        #Loss function
        self.l_adv = l2_loss
        self.l_con = nn.L1Loss()
        self.l_enc = l2_loss
        self.l_bce = nn.BCELoss()

        #Initialize input tensors
        self.input = torch.empty(size=inp_shape, dtype=torch.float32, device=self.device)
        self.label = torch.empty(size=inp_shape, dtype=torch.float32, device=self.device)
        self.gt = torch.empty(size=inp_shape, dtype=torch.float32, device=self.device)
        self.fixed_input = torch.empty(size=inp_shape, dtype=torch.float32, device=self.device)
        self.real_label = torch.ones(size=(self.args.batchsize,), 
                                            dtype=torch.float32, device=self.device)
        self.gout_label = torch.zeros(size=(self.args.batchsize,), 
                                            dtype=torch.float32, device=self.device)

        #Setup Optimizer
        if self.args.isTrain:
            self.netg.train()
            self.netd.train()
            self.optimizer_d = optim.Adam(self.netd.parameters(), 
                                    lr= self.args.lr, betas=(self.args.beta1, 0.999))
            self.optimizer_g = optim.Adam(self.netg.parameters(), 
                                    lr= self.args.lr, betas=(self.args.beta1, 0.999))
        

    def forward_g(self):
        self.gout = self.netg(self.input)

    def forward_d(self):
        self.input_flow = video_to_flow(self.input).to(self.device)
        self.gout_flow = video_to_flow(self.gout.detach()).to(self.device)
        self.s_pred_real, self.s_feat_real, self.t_pred_real, self.t_feat_real \
                = self.netd(self.input, self.input_flow.detach())
        self.s_pred_fake, self.s_feat_fake, self.t_pred_fake, self.t_feat_fake \
                = self.netd(self.gout.detach(), self.gout_flow.detach())
    
    def backward_g(self):
        predict = predict_forg(self.input.detach(), self.gout.detach())
        predict = torch.from_numpy(predict).to(self.device)
        self.err_g_adv_s = self.l_adv(self.s_feat_real, self.s_feat_fake)
        self.err_g_adv_t = self.l_adv(self.t_feat_real, self.t_feat_fake)
        self.err_g_adv = self.err_g_adv_s + self.err_g_adv_t
        self.err_g_con = self.l_con(self.gout, self.input)
        self.err_g_pre = self.l_bce(predict, self.gt)
        self.err_g = self.err_g_adv * self.args.w_adv + \
                    self.err_g_con * self.args.w_con + \
                    self.err_g_pre * self.args.w_pre
        self.err_g.backward(retain_graph=True)

    def backward_d(self):
        self.err_d_real_s = self.l_bce(self.s_pred_real, self.real_label)
        self.err_d_real_t = self.l_bce(self.t_pred_real, self.real_label)
        self.err_d_fake_s = self.l_bce(self.s_pred_fake, self.gout_label)
        self.err_d_fake_t = self.l_bce(self.t_pred_fake, self.gout_label)

        self.err_d_real = (self.err_d_real_s + self.err_d_real_t) * 0.5
        self.err_d_fake = (self.err_d_fake_s + self.err_d_fake_t) * 0.5

        self.err_d = (self.err_d_real + self.err_d_fake) * 0.5
        self.err_d.backward()

    def reinit_d(self):
        self.netd.apply(weights_init)
        print('Reloading Net d')

    def optimize_params(self):

        self.forward_g()
        self.forward_d()

        #netg
        self.optimizer_g.zero_grad()
        self.backward_g()
        self.optimizer_g.step()

        #netd
        self.optimizer_d.zero_grad()
        self.backward_d()
        self.optimizer_d.step()
        if self.err_d.item() < 1e-5: self.reinit_d()



