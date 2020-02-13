import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from lib.train_gan import GANBaseModel
from lib.evaluate import evaluate
from lib.utils import *
from models.spatiotempconv import SpatioTemporalConv

class NetgConv(nn.Module):
    def __init__(self, in_fi, out_fi, kernel_size=3):
        super(NetgConv, self).__init__()
        padding = kernel_size//2
        self.conv = SpatioTemporalConv(in_fi, out_fi, kernel_size, padding=padding)
        #self.conv = nn.Conv3d(in_fi, out_fi, 3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(out_fi)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)

        return x


class NetG(nn.Module):
    def __init__(self, nc=3, ngf=32):
        super(NetG, self).__init__()

        self.dconv1 = NetgConv(nc, ngf)
        self.dconv2 = NetgConv(ngf, ngf*2)
        self.dconv3 = NetgConv(ngf*2, ngf*4)
        self.dconv4 = NetgConv(ngf*4, ngf*8)
        self.dconv5 = NetgConv(ngf*8, ngf*16)

        self.avgpool = nn.AvgPool3d(2)
        
        self.uconv5 = NetgConv(ngf*16, ngf*8)
        self.uconv4 = NetgConv(ngf*8+ngf*8, ngf*8)
        self.uconv3 = NetgConv(ngf*8+ngf*4, ngf*4)
        self.uconv2 = NetgConv(ngf*4+ngf*2, ngf*2)
        self.uconv1 = NetgConv(ngf*2+ngf, ngf)
        
        self.dropout = nn.Dropout(p=0.25)
        self.upsamp = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.conv_last = nn.Conv3d(ngf, 1, 3, stride=1, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # Encode 1
        # (32, 128)
        dconv1 = self.dconv1(x) # ngf
        x = self.avgpool(dconv1)
        # (16, 64)
        dconv2 = self.dconv2(x) # ngf*2
        x = self.avgpool(dconv2)
        # (8, 32)
        dconv3 = self.dconv3(x) # ngf*4
        x = self.avgpool(dconv3)
        # (4, 16)
        dconv4 = self.dconv4(x) # ngf*8
        x = self.avgpool(dconv4)
        # (2, 8)

        latent_i = self.dconv5(x) # ngf*16

        #Decoder
        x = self.uconv5(latent_i) # ngf*8
        x = self.dropout(x)
        x = self.upsamp(x) # ngf*8
        # (4, 8)
        x = torch.cat([x, dconv4], dim=1) # ngf*8*2
        x = self.uconv4(x) # ngf*8
        x = self.dropout(x)
        x = self.upsamp(x)
        # (4, 16)
        x = torch.cat([x, dconv3], dim=1) #ngf*8+ngf*4
        x = self.uconv3(x)
        x = self.dropout(x)
        x = self.upsamp(x)
        # (8, 32)
        x = torch.cat([x, dconv2], dim=1)
        x = self.uconv2(x)
        x = self.dropout(x)
        x = self.upsamp(x)
        # (16, 64)
        x = torch.cat([x, dconv1], dim=1)
        x = self.uconv1(x)

        predict = self.conv_last(x)

        predict = self.sigmoid(predict)

        return predict
        

class NetdConv(nn.Module):
    def __init__(self, in_fi, out_fi, kernel_size=None, padding=None):
        super(NetdConv, self).__init__()
        self.conv = SpatioTemporalConv(in_fi, out_fi, kernel_size, padding=padding)
        #self.conv = nn.Conv3d(in_fi, out_fi, kernel, stride=1, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_fi)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)
        return x


class SDisc(nn.Module):
    def __init__(self, nc, nfr, ndf=32, kernel=None, padding=None):
        super(SDisc, self).__init__()

        # input size == (B, C, D, H, W)
        netdconv = lambda in_fi, out_fi: NetdConv(in_fi, out_fi, kernel_size=kernel, padding=padding)
        self.dconv1 = netdconv(nc, ndf)
        self.dconv2 = netdconv(ndf, ndf*2)
        self.dconv3 = netdconv(ndf*2, ndf*4)
        self.dconv4 = netdconv(ndf*4, ndf*8)
        self.dconv5 = netdconv(ndf*8, ndf*16)
        self.dconv6 = netdconv(ndf*16, ndf*32)

        self.avgpool = nn.AvgPool3d((1, 2, 2)) 
        self.gpool = nn.AvgPool3d((nfr, 1, 1), stride=1)
        self.linear = nn.Linear(ndf*32*2*2, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # (32, 128)
        x = self.dconv1(x)
        x = self.avgpool(x)
        # (32, 64)
        x = self.dconv2(x)
        x = self.avgpool(x)
        # (32, 32)
        x = self.dconv3(x)
        x = self.avgpool(x)
        # (32, 16)
        x = self.dconv4(x)
        x = self.avgpool(x)
        # (32, 8)
        x = self.dconv5(x)
        x = self.avgpool(x)
        # (32, 4)
        x = self.dconv6(x)
        features = self.avgpool(x)
        # (32, 2)
        x = self.gpool(features) # nfr -> 1
        x = self.linear(x.view(x.shape[0], -1))
        classifier = self.sigmoid(x)

        return classifier.squeeze(1), features

class TDisc(nn.Module):
    def __init__(self, nc, isize, ndf=32, kernel=None, padding=None):
        super(TDisc, self).__init__()

        # input size == (B, C, D, H, W)
        netdconv = lambda in_fi, out_fi: NetdConv(in_fi, out_fi, kernel_size=kernel, padding=padding)
        self.dconv1 = netdconv(nc, ndf)
        self.dconv2 = netdconv(ndf, ndf*2)
        self.dconv3 = netdconv(ndf*2, ndf*4)

        self.avgpool = nn.AvgPool3d((2, 1, 1)) 
        self.gpool = nn.AvgPool3d((1, isize, isize), stride=1)
        self.linear = nn.Linear(ndf*4*2, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # (16, 128)
        x = self.dconv1(x)
        x = self.avgpool(x)
        # (8, 128)
        x = self.dconv2(x)
        x = self.avgpool(x)
        # (4, 32)
        x = self.dconv3(x)
        features = self.avgpool(x)
        # (2, 16)

        x = self.gpool(features) # isize -> 1
        x = self.linear(x.view(x.shape[0], -1)) # (B, ndf*4, 2, 1, 1) -> (B, 1)
        classifier = self.sigmoid(x)

        return classifier.squeeze(1), features



class NetD(nn.Module):
    def __init__(self, args):
        super(NetD, self).__init__()

        self.spatdisc = SDisc(3, args.nfr, kernel=(1, 3, 3), padding=(0, 1, 1))
        self.tempdisc = TDisc(3, args.isize, kernel=(3, 1, 1), padding=(1, 0, 0))
        #self.tempdisc = SDisc(3, args.nfr, kernel=(1, 3, 3), padding=(0, 1, 1))

    def forward(self, x, y):

        s_cls, s_feat = self.spatdisc(x)
        t_cls, t_feat = self.tempdisc(y)

        return s_cls, s_feat, t_cls, t_feat
        

class MyGAN(GANBaseModel):
    def __init__(self, args, dataloader):
        super(MyGAN, self).__init__(args, dataloader)

        inp_shape = (self.args.batchsize, self.args.ich, 
                    self.args.nfr, self.args.isize, self.args.isize)

        netd = NetD
        if self.args.ae:
            print("\n --Load C2plus1d Model as G -- ")
            from models.mystcnn import AutoEncoder
            netg = AutoEncoder()
        else:
            print("\n --Load Normal Model as G -- ")
            netg = NetG

        if len(self.args.gpu) > 1:
            self.netg = torch.nn.DataParallel(netg(), device_ids=self.args.gpu, dim=0)
            self.netd = torch.nn.DataParallel(netd(self.args), device_ids=self.args.gpu, dim=0)
            self.netg = self.netg.cuda()
            self.netd = self.netd.cuda()
            cudnn.benchmark = True
        else:
            self.netg = netg().to('cuda')
            self.netd = netd(self.args).to('cuda')
        self.netg.apply(weights_init)
        self.netd.apply(weights_init)
      
        # Load pre-trained network weight
        if self.args.resume != '':
            print("\n Loading pretrained network weight = {}".format(self.args.resume))
            G_resume = self.args.resume
            D_resume = os.path.join(self.args.resume.rsplit("_", 1)[0] + "netD.pth")
            G_state_dict = torch.load(G_resume, map_location='cuda:0')['state_dict']
            D_state_dict = torch.load(D_resume, map_location='cuda:0')['state_dict']
            try:
                self.netg.load_state_dict(fix_model_state_dict(G_state_dict))
                self.netd.load_state_dict(fix_model_state_dict(D_state_dict))
            except IOError:
                raise IOError("Model weights not found")
            print("\n Done.\n")

        self.real_label = torch.ones(size=(self.args.batchsize,), 
                                        dtype=torch.float32, device='cuda')
        self.gout_label = torch.zeros(size=(self.args.batchsize,), 
                                        dtype=torch.float32, device='cuda')

        #Loss function
        self.l_adv = l2_loss
        self.l_con = lambda output, target: weighted_bce(output, target, pos_weight=self.args.pos_weight)
        self.l_con = weighted_bce
        self.l_bce = nn.BCELoss()

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
        gt_flow = video_to_flow(gt_3ch.detach()).to('cuda')
        pre_flow = video_to_flow(pre_3ch.detach()).to('cuda')
        self.s_pred_real, self.s_feat_real, self.t_pred_real, self.t_feat_real \
                = self.netd(gt_3ch, gt_flow.detach())
        self.s_pred_fake, self.s_feat_fake, self.t_pred_fake, self.t_feat_fake \
                = self.netd(pre_3ch.detach(), pre_flow.detach())

        t_pre = threshold(self.predict.detach())
        m_pre = morphology_proc(t_pre.detach())
        
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

        self.netg.train()
        self.netd.train()

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

    def test(self):
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
            # Test
            pbar = tqdm(self.dataloader['test'], leave=True, ncols=100, total=len(self.dataloader['test']))
            for i, data in enumerate(pbar):
                # set test data 
                input, real, gt, lb = (d.to('cuda') for d in data)
                # NetG
                predict_ = self.netg(input) # Reconstract self.input
                t_pre_ = threshold(predict_.detach())
                m_pre_ = morphology_proc(t_pre_)
                gts.append(gt.permute(0, 2, 3, 4, 1).cpu().numpy())
                predicts.append(m_pre_.permute(0, 2, 3, 4, 1).cpu().numpy())
                # NetD
                # calc Optical Flow 
                gt_3ch_ = gray2rgb(gt)
                pre_3ch_ = gray2rgb(predict_)
                gt_flow_ = video_to_flow(gt_3ch_.detach()).to('cuda')
                pre_flow_ = video_to_flow(pre_3ch_).to('cuda')
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
                err_g_.append(err_g_adv_t_[-1] * self.args.w_adv + err_g_con_[-1] * self.args.w_con)
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

