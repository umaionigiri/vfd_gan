
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

from network import NetD
from evaluate import evaluate
from utils import *

class BaseModel():
    """
    Base Model for GAN

    """

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
        
        # set using gpu
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    def update_summary(self):
        # VIDEO
        for t, v in self.color_video_dict.items():
            grid = [make_grid(f, nrow=self.args.batchsize, normalize=True) for f in v.permute(2, 0, 1, 3, 4)]
            self.writer.add_video(t, torch.unsqueeze(torch.stack(grid), 0), self.global_step)

        for t, v in self.gray_video_dict.items():
            grid = [make_grid(f, nrow=self.args.batchsize, normalize=False) for f in v.permute(2, 0, 1, 3, 4)]
            self.writer.add_video(t, torch.unsqueeze(torch.stack(grid), 0), self.global_step)

        # ERROR
        for t, e in self.errors_dict.items():
            spk = t.rsplit('/',1)
            self.writer.add_scalars(spk[0], {spk[1]: e}, self.global_step)
        """
        # HISTOGRAM
        for t, h in self.hist_dict.items():
            self.writer.add_histogram(t, h)
        """

        # SCORE
        for t, s in self.score_dict.items():
            self.writer.add_scalar(t, s, self.global_step)

    def save_weights(self, name_head):
        # -- Save Weights of NetG and NetD --
        torch.save({'epoch': self.epoch + 1, 'state_dict': self.netg.state_dict()},
                   '%s/%s_ep%04d_netG.pth' % (self.weight_dir, name_head, self.epoch))
        torch.save({'epoch': self.epoch + 1, 'state_dict': self.netd.state_dict()},
                   '%s/%s_ep%04d_netD.pth' % (self.weight_dir, name_head, self.epoch))


    def test_epoch_end(self):
        
        self.netg.eval()
        self.netd.eval()

        err_g_adv_s_ = []
        err_g_adv_t_ = []
        err_g_adv_ = []
        err_g_con_ = []
        err_g_pre_ = []
        err_g_ = []

        err_d_real_s_ = []
        err_d_real_t_ = []
        err_d_fake_s_ = []
        err_d_fake_t_ = []
        err_d_real_ = []
        err_d_fake_ = []
        err_d_ = []

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
                # set test data 
                input, real, gt, lb = (d.to(self.device) for d in data)
                # NetG
                predict_ = self.netg(input) # Reconstract self.input
                t_pre_ = threshold(predict_.detach())
                m_pre_ = morphology_proc(t_pre_)
                gts.append(gt.permute(0, 2, 3, 4, 1).cpu().numpy())
                predicts.append(predict_.permute(0, 2, 3, 4, 1).cpu().numpy())
                # NetD
                # calc Optical Flow 
                gt_3ch_ = gray2rgb(gt)
                pre_3ch_ = gray2rgb(predict_)
                gt_flow_ = video_to_flow(gt_3ch_.detach()).to(self.device)
                pre_flow_ = video_to_flow(pre_3ch_).to(self.device)
                # get disc output
                s_pred_real_, s_feat_real_, t_pred_real_, t_feat_real_ \
                                        = self.netd(gt_3ch_, gt_flow_.detach())
                s_pred_fake_, s_feat_fake_, t_pred_fake_, t_feat_fake_ \
                                        = self.netd(pre_3ch_.detach(), pre_flow_.detach())
                # Calc err_g
                err_g_adv_s_.append(self.l_adv(s_feat_real_, s_feat_fake_).item())
                err_g_adv_t_.append(self.l_adv(t_feat_real_, t_feat_fake_).item())
                err_g_adv_.append(err_g_adv_s_[-1] + err_g_adv_t_[-1])
                err_g_con_.append(self.l_con(predict_, gt).item())
                err_g_con_.append(err_g_adv_t_[-1] * self.args.w_adv + err_g_con_[-1] * self.args.w_con)
                # Calc err_d
                err_d_real_s_.append(self.l_bce(s_pred_real_, self.real_label).item())
                err_d_real_t_.append(self.l_bce(t_pred_real_, self.real_label).item())
                err_d_fake_s_.append(self.l_bce(s_pred_fake_, self.gout_label).item())
                err_d_fake_t_.append(self.l_bce(t_pred_fake_, self.gout_label).item())
                err_d_real_.append((err_d_real_s_[-1] + err_d_real_t_[-1]) * 0.5)
                err_d_fake_.append((err_d_fake_s_[-1] + err_d_fake_t_[-1]) * 0.5)
                err_d_.append((err_d_real_[-1] + err_d_fake_[-1]) * 0.5)
                
                # test video summary
                self.color_video_dict.update({
                        'test/input-real': torch.cat([input, real], dim=3),
                    })
                self.gray_video_dict.update({
                        'test/gt-pre-th-morph': torch.cat([gt, predict_, t_pre_, m_pre_], dim=3)
                    })
                self.hist_dict.update({
                    "test/inp": input,
                    "test/gt": gt,
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
                        'd/err_d_real_s/test': np.mean(err_d_real_s_),
                        'd/err_d_real_t/test': np.mean(err_d_real_t_),
                        'd/err_d_fake_s/test': np.mean(err_d_fake_s_),
                        'd/err_d_fake_t/test': np.mean(err_d_fake_t_),
                        'd/err_d_real/test': np.mean(err_d_real_),
                        'd/err_d_fake/test': np.mean(err_d_fake_),
                        'd/err_d/test': np.mean(err_d_),
                        'g/err_g_adv_s/test': np.mean(err_g_adv_s_),
                        'g/err_g_adv_t/test': np.mean(err_g_adv_t_),
                        'g/err_g_adv/test': np.mean(err_g_adv_),
                        'g/err_g_con/test': np.mean(err_g_con_),
                        'g/err_g/test': np.mean(err_g_)
                        })

    def train(self):

        print(" >> Training model %s." % self.args.model)

        for self.epoch in range(self.args.ep):
            pbar = tqdm(self.dataloader['train'], leave=True, ncols=100, total=len(self.dataloader['train']))
            for i, data in enumerate(pbar):

                self.global_step += 1
                self.netg.train()
                self.netd.train()

                self.input, self.real, self.gt, self.lb = (d.to(self.device) for d in data)
                self.optimize_params()

                if self.global_step % self.args.test_freq == 0:
                    # -- TEST --
                    self.test_epoch_end()
                
                if self.global_step % self.args.display_freq == 0:
                    # -- Update Summary on Tensorboard --
                    self.update_summary()

                pbar.set_description("[TRAIN Epoch %d/%d]" % (self.epoch+1, self.args.ep))
     
            print(">> Training model %s. Epoch %d/%d " % (self.name, self.epoch+1, self.args.ep))
            
        print(" >> Training model %s.[Done]" % self.args.model)



class VFD_GAN(BaseModel):

    @property
    def name(self): return 'VFD_GAN'

    def __init__(self, args, dataloader):
        super(VFD_GAN, self).__init__(args, dataloader)


        inp_shape = (self.args.batchsize, self.args.ich, 
                    self.args.nfr, self.args.isize, self.args.isize)

        if self.args.ae:
            from vfd_c2plus1d import AutoEncoder
            Generator = AutoEncoder()
        else:
            from network import NetG
            Generator = NetG()


        # Create and initialize networkgs.
        if len(self.args.gpu) > 1:
            self.netg = torch.nn.DataParallel(Generator, device_ids=self.args.gpu, dim=0)
            self.netd = torch.nn.DataParallel(NetD(self.args), device_ids=self.args.gpu, dim=0)
            self.netg = self.netg.cuda()
            self.netd = self.netd.cuda()
            cudnn.benchmark = True
        else:
            self.netg = Generator.to(self.device)
            self.netd = NetD(self.args).to(self.device)
        self.netg.apply(weights_init)
        self.netd.apply(weights_init)

        #pre-trained network load
        if self.args.resume != '' :
            print("\n Loading pre-trained networks.")
            #self.args.iter = torch.load(self.args.resume)['epoch']
            self.netg.load_state_dict(torch.load(os.path.join(self.args.resume, \
                                                            'ep0004_netG.pth'))['state_dict'])
            self.netd.load_state_dict(torch.load(os.path.join(self.args.resume, 
                                                            'ep0004_netD.pth'))['state_dict'])
            print("\t Done. \n")
        
        self.real_label = torch.ones(size=(self.args.batchsize,), dtype=torch.float32, device=self.device)
        self.gout_label = torch.zeros(size=(self.args.batchsize,), dtype=torch.float32, device=self.device)

        #Loss function
        self.l_adv = l2_loss
        #self.l_pre = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(30))
        self.l_con = nn.L1Loss()
        self.l_bce = bce_smooth

        #Setup Optimizer
        self.optimizer_d = optim.Adam(self.netd.parameters(), 
                                lr= self.args.lr, betas=(self.args.beta1, 0.999))
        self.optimizer_g = optim.Adam(self.netg.parameters(), 
                                lr= self.args.lr, betas=(self.args.beta1, 0.999))

    def forward_g(self):
        self.predict = self.netg(self.input)
        
    def forward_d(self):
        pre_3ch = gray2rgb(self.predict.detach())
        gt_3ch = gray2rgb(self.gt.detach())
        gt_flow = video_to_flow(gt_3ch.detach()).to(self.device)
        pre_flow = video_to_flow(pre_3ch.detach()).to(self.device)
        self.s_pred_real, self.s_feat_real, self.t_pred_real, self.t_feat_real \
                = self.netd(gt_3ch, gt_flow.detach())
        self.s_pred_fake, self.s_feat_fake, self.t_pred_fake, self.t_feat_fake \
                = self.netd(pre_3ch.detach(), pre_flow.detach())

        t_pre = threshold(self.predict.detach())
        m_pre = morphology_proc(self.predict.detach())
        
        # set dict for summary
        self.color_video_dict.update({
                "train/input-real-inflow-genflow": torch.cat([self.input, self.real, gt_flow, pre_flow], dim=3)})
        self.gray_video_dict.update({
                "train/gt-pre-th-morph": torch.cat([self.gt, self.predict, t_pre, m_pre], dim=3)
                })
        self.hist_dict.update({
            "train/input": self.input,
            "train/gt": self.gt,
            "train/predict": self.predict,
            "train/t_pre": t_pre,
            "train/m_pre": m_pre
            })
    
    def backward_g(self):
        err_g_adv_s = self.l_adv(self.s_feat_real, self.s_feat_fake)
        err_g_adv_t = self.l_adv(self.t_feat_real, self.t_feat_fake)
        err_g_adv = err_g_adv_s + err_g_adv_t
        err_g_con = self.l_con(self.predict, self.gt)
        err_g = err_g_adv * self.args.w_adv + \
                     err_g_con * self.args.w_con
        err_g.backward(retain_graph=True)

        self.errors_dict.update({
                        'g/err_g/train': err_g.item(),
                        'g/err_g_adv/train': err_g_adv.item(),
                        'g/err_g_adv_s/train': err_g_adv_s.item(),
                        'g/err_g_adv_t/train': err_g_adv_t.item(),
                        'g/err_g_con/train': err_g_con.item()
                        })


    def backward_d(self):
        err_d_real_s = self.l_bce(self.s_pred_real, self.real_label)
        err_d_real_t = self.l_bce(self.t_pred_real, self.real_label)
        err_d_fake_s = self.l_bce(self.s_pred_fake, self.gout_label)
        err_d_fake_t = self.l_bce(self.t_pred_fake, self.gout_label)

        err_d_real = (err_d_real_s + err_d_real_t) * 0.5
        err_d_fake = (err_d_fake_s + err_d_fake_t) * 0.5

        err_d = (err_d_real + err_d_fake) * 0.5

        self.errors_dict.update({
                        'd/err_d_real_s/train': err_d_real_s.item(),
                        'd/err_d_real_t/train': err_d_real_t.item(),
                        'd/err_d_fake_s/train': err_d_fake_s.item(),
                        'd/err_d_fake_t/train': err_d_fake_t.item(),
                        'd/err_d_real/train': err_d_real.item(),
                        'd/err_d_fake/train': err_d_fake.item(),
                        'd/err_d/train': err_d.item()
                        })

        err_d.backward()

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
        #if self.err_d.item() < 1e-5: self.reinit_d()



