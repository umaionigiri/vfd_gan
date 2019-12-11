
import time
import os
import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np

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
from utils import video_to_flow


def l2_loss(inp, target, size_average=True):
    if size_average:
        return torch.mean(torch.pow((inp-target), 2))
    else:
        return torch.pow((inp-target), 2)

class BaseModel():
    """
    Base Model for ganomaly

    """

    def __init__(self, args, dataloader):

        self.args = args
        self.dataloader = dataloader

        self.imgs_dict = OrderedDict()
        self.errors_dict = OrderedDict()
        
        # set using gpu
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # make save root dir
        comment = "bs-{}_lr-{}_w-a{}c{}e{}".format(args.batchsize, args.lr, 
                                                    args.w_adv, args.w_con, args.w_enc)
        self.save_root_dir = os.path.join(args.result_root, args.model, comment)
        if not os.path.exists(self.save_root_dir): os.makedirs(self.save_root_dir)
        # make weights save dir
        self.weight_dir = os.path.join(self.save_root_dir,'weights')
        if not os.path.exists(self.weight_dir): os.makedirs(self.weight_dir)
        # make tensorboard logdir 
        from datetime import datetime
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        logdir = os.path.join(self.save_root_dir, "runs", current_time)
        if not os.path.exists(logdir): os.makedirs(logdir)
        self.writer = SummaryWriter(log_dir=logdir)
        #self.writer = SummaryWriter(comment=comment)
                
        
    def set_input(self, input:torch.Tensor):
        with torch.no_grad():
            self.input.resize_(input[0].size()).copy_(input[0])
            self.gt.resize_(input[2].size()).copy_(input[2])
            self.label.resize_(input[2].size())
            if self.total_steps == self.args.batchsize:
                self.fixed_input.resize_(input[0].size()).copy_(input[0])


    def get_errors(self):
        self.errors_dict.update({
                        ('err_d', self.err_d.item()),
                        ('err_g', self.err_g.item()),
                        ('err_g_adv', self.err_g_adv.item()),
                        ('err_g_con', self.err_g_con.item()),
                        #('err_g_enc', self.err_g_enc.item()),
                        })

    def get_train_images(self):

        reals = self.input.data
        fakes = self.fake
        fixed = self.netg(self.fixed_input)[0]
        
        self.imgs_dict.update({
                "train input/gen": torch.cat([reals, fakes], dim=3),
            })
        
    def update_summary(self):
        self.get_errors()
        self.get_train_images()
        for tag, err in self.errors_dict.items():
            self.writer.add_scalar(tag, err, self.total_steps)
        for tag, v in self.imgs_dict.items():
            grid = [make_grid(f, nrow=self.args.batchsize, 
                            normalize=True) 
                    for f in v.permute(2, 0, 1, 3, 4)]
            self.writer.add_video(tag, torch.unsqueeze(torch.stack(grid), 0), self.total_steps)

    def save_weights(self, epoch):
        torch.save({'epoch': epoch + 1, 'state_dict': self.netg.state_dict()},
                   '%s/ep%04d_netG.pth' % (self.weight_dir, epoch))
        torch.save({'epoch': epoch + 1, 'state_dict': self.netd.state_dict()},
                   '%s/ep%04d_netD.pth' % (self.weight_dir, epoch))


    def train(self):

        self.total_steps = 0
        best_auc = 0
        epoch_iter = 0
        
        
        print(" >> Training model %s." % self.args.model)

        for self.epoch in range(self.args.ep):

            self.netg.train()
            pbar = tqdm(self.dataloader['train'], leave=True, 
                        ncols=100, total=len(self.dataloader['train']))
            for i, data in enumerate(pbar):
                self.total_steps += 1
                epoch_iter += 1

                self.set_input(data)
                self.optimize_params()
                
                if self.total_steps % self.args.display_freq == 0:
                    print("\n--update summary--")
                    self.update_summary()
                
                pbar.set_postfix(OrderedDict(loss="{:.4f}".format(self.err_g)))
                pbar.set_description("[Epoch %d/%d]" % (self.epoch+1, self.args.ep))

            #Test
            """
            res = self.test()
            if res['AUC'] > best_auc:
                best_auc = res['AUC']
            """
            if self.epoch % self.args.save_weight_freq == 0:
                self.save_weights(self.epoch)

            print(">> Training model %s. Epoch %d/%d " % (self.name, self.epoch+1, self.args.ep))
            
        print(" >> Training model %s.[Done]" % self.args.model)

    def test(self):

        with torch.no_grad():
            if self.args.load_weights != " " :
                #path = self.weight_dir + "ep%04d_netG.pth".format(self.epoch)
                path = self.args.load_weights
                pretrained_dict = torch.load(path)['state_dict']

                try:
                    self.netg.load_state_dict(pretrained_dict)
                except IOError:
                    raise IOError("netG weights not found")
                print("Loaded weights.")

            self.args.phase = 'test'

            """
            self.an_scores = torch.zeros(size=(len(self.dataloader['test'].dataset),), 
                                                dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(self.dataloader['test'].dataset),), 
                                                dtype=torch.long, device=self.device)
            self.latent_i = torch.zeros(size=(len(self.dataloader['test'].dataset), self.args.nz),
                                                dtype=torch.float32, device=self.device)
            self.latent_o = torch.zeros(size=(len(self.dataloader['test'].dataset), self.args.nz),
                                                dtype=torch.float32, device=self.device)
            """
            

            self.times = []
            self.total_steps = 0
            epoch_iter = 0
            
            self.gt_allfr = []
            self.pre_allfr = []
            __fgp = []

            for i, data in enumerate(self.dataloader['test'], 0):
                self.total_steps += self.args.batchsize
                epoch_iter += 1
                time_i = time.time()
                predict = []

                #Initialize data 
                self.set_input(data)
                self.fake, latent_i, latent_o = self.netg(self.input)
                self.gt = self.gt
                self.input = self.input
                self.fake = self.fake
                print(self.fake.shape)

                #Tensor(RGB) to Numpy(Gray)
                input = self.tensor_to_ndarray(self.input, gray=False)
                grinp = self.tensor_to_ndarray(self.input)
                grfake = self.tensor_to_ndarray(self.fake)
                gt = self.tensor_to_ndarray(self.gt, gray=False)
                #vutils.save_image(torch.from_numpy(grinp)[0][0], '/mnt/fs2_rwx/2018/ohshiro/grinp.png')
                #Difference between input to fake
                _predicts = []
                for inp, fake, g in zip(grinp, grfake, gt):
                    predicts = []
                    for _inp, _fake, _g in zip(inp, fake, g):
                        predict= np.array(_inp - _fake > 127, dtype=np.int32)
                        ground = np.array(_g > 127, dtype=np.int32)
                        self.pre_allfr.append(predict)
                        self.gt_allfr.append(ground)
                        predicts.append(predict)

                    _predicts.append(np.stack(predicts))

                self.writer.add_video('test_input', self.input, epoch_iter)
                self.writer.add_video('test_fake', self.fake, epoch_iter)
                self.writer.add_video('test_gt', self.gt, epoch_iter)
                self.writer.add_video('test_predict',np.stack(_predicts), epoch_iter)



                #vutils.save_image(torch.from_numpy(predicts)[0][0], '/mnt/fs2_rwx/2018/ohshiro/predicts.png')

                time_o = time.time()
                self.times.append(time_o - time_i)


                """
                error = torch.mean(torch.pow((latent_i-latent_o), 2), dim=1)
                print("\n error shape == {}".format(error.size(0)))
                print("\n gt labels shape == {}".format(self.gt_labels.shape))
                print("\n gt shape == {}".format(self.gt.shape))
                time_o = time.time()

                self.an_scores[i*self.args.batchsize : i*self.args.batchsize+error.size(0)] \
                        = error.reshape(error.size(0))
                self.gt_labels[i*self.args.batchsize : i*self.args.batchsize+error.size(0)] \
                        = self.gt.reshape(error.size(0))
                self.latent_i[i*self.args.batchsize : i*self.args.batchsize+error.size(0), :] \
                        = latent_i.reshape(error.size(0), self.args.nz)
                self.latent_o[i*self.args.batchsize : i*self.args.batchsize+error.size(0), :] \
                        = latent_o.reshape(error.size(0), self.args.nz)
                if self.args.save_test_images:
                    dst = os.path.join(self.args.result_root, self.args.model, 'test', 'images')
                    if not os.path.isdir(dst):
                        os.makedirs(dst)
                    real, fake, _ = self.get_current_images()
                    vutils.save_image(real, '%s/real_%03d.eps' % (dst, i+1), normalize=True)
                    vutils.save_image(fake, '%s/fake_%03d.eps' % (dst, i+1), normalize=True)
                """

           
           
            #vutils.save_image(torch.from_numpy(self.pre_allfr), '/mnt/fs2_rwx/2018/ohshiro/pre_allfr.png')
            
            self.pre_allfr = np.asarray(self.pre_allfr).flatten()
            self.gt_allfr = np.asarray(self.gt_allfr).flatten()
            auc = evaluate(self.gt_allfr, self.pre_allfr, self.tst_dir,  metric=self.args.metric)

            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)

            performance = OrderedDict([('Avg Run Time (ms / batch)', self.times), ('AUC', auc)])


            """
            self.an_scores = (self.an_scores - torch,min(self.an_scores)) \
                            / (torch.max(self.an_scores) - torch.min(self.an_scores))
            auc = evaluate(self.gt_labels, self.an_scores, metric=self.args.metric)
            """

            return performance


class Ganomaly(BaseModel):

    @property
    def name(self): return 'Ganomaly'

    def __init__(self, args, dataloader):
        super(Ganomaly, self).__init__(args, dataloader)

        # -- Misc attributes
        self.epoch = 0
        self.times = []

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
        self.fake_label = torch.zeros(size=(self.args.batchsize,), 
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
        self.latent_i, self.fake = self.netg(self.input)


    def forward_d(self):
        self.input_flow = video_to_flow(self.input).to(self.device)
        self.fake_flow = video_to_flow(self.fake.detach()).to(self.device)
        self.s_pred_real, self.s_feat_real, self.t_pred_real, self.t_feat_real \
                = self.netd(self.input, self.input_flow.detach())
        self.s_pred_fake, self.s_feat_fake, self.t_pred_fake, self.t_feat_fake \
                = self.netd(self.fake.detach(), self.fake_flow.detach())
    
    def backward_g(self):
        self.err_g_adv_s = self.l_adv(self.s_feat_real, self.s_feat_fake)
        self.err_g_adv_t = self.l_adv(self.t_feat_real, self.t_feat_fake)
        self.err_g_adv = self.err_g_adv_s + self.err_g_adv_t
        self.err_g_con = self.l_con(self.fake, self.input)
        #self.err_g_enc = self.l_enc(self.latent_o, self.latent_i)
        self.err_g = self.err_g_adv * self.args.w_adv + \
                    self.err_g_con * self.args.w_con  \
                    #self.err_g_enc * self.args.w_enc
        self.err_g.backward(retain_graph=True)

    def backward_d(self):
        self.err_d_real_s = self.l_bce(self.s_pred_real, self.real_label)
        self.err_d_real_t = self.l_bce(self.t_pred_real, self.real_label)
        self.err_d_fake_s = self.l_bce(self.s_pred_fake, self.fake_label)
        self.err_d_fake_t = self.l_bce(self.t_pred_fake, self.fake_label)

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



