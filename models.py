
import numpy as np
from tqdm import tqdm
import os
import json
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
from utils import weights_init

class SpatialTempModel():
    def __init__(self, args, dataloader):

        self.args = args
        self.dataloader = dataloader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if args.model == "xception":
            print("\n --Load Xception Model-- ")
            from xception import Xception
            model = Xception()
        elif args.model == "clstm":
            print("\n --Load ConvLSTM Model-- ")
            from convlstm import ConvLSTMModel
            model = ConvLSTMModel(args)
        elif args.model == "c2plus1d":
            print("\n --Load C2plus1d Model-- ")
            from vfd_c2plus1d import AutoEncoder
            model = AutoEncoder()
        else:
            print("\n Model name is wrong \n")

        if len(self.args.gpu) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=self.args.gpu, dim=0)
            self.model = self.model.cuda()
            cudnn.benchmark = True
        else:
            self.model = model.to(self.device)
        self.model.apply(weights_init)

        # Load pre-trained network weight
        if self.args.resume != '':
            print("\n Loading pretrained network weight = {}".format(self.args.resume))
            self.model.load_state_dict(torch.load(os.path.join(self.args.resume, 
                                                                'ep00.pth'))['state_dict'])
            print("\n Done.\n")

        # Initialize setting
        self.global_step = 0
        self.best_roc = 0
        self.best_pr = 0
        self.best_f1 = 0
        self.color_video_dict = OrderedDict()
        self.gray_video_dict = OrderedDict()
        self.train_errors_dict = OrderedDict()
        self.test_errors_dict = OrderedDict()
        self.score_dict = OrderedDict()

        # Make save dir for weight, tensorboard run file
        from datetime import datetime
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        comment = "b{}xd{}xwh{}_lr{}".format(args.batchsize, args.nfr, args.isize, args.lr)
        self.save_root_dir = os.path.join(args.result_root, args.model, comment, current_time)
        if not os.path.exists(self.save_root_dir): os.makedirs(self.save_root_dir)
        # make weights save dir
        self.weight_dir = os.path.join(self.save_root_dir, 'weights')
        if not os.path.exists(self.weight_dir): os.makedirs(self.weight_dir)
        # make tensorboard logdir
        logdir = os.path.join(self.save_root_dir, "runs")
        self.writer = SummaryWriter(log_dir=logdir)
        # save args 
        with open(self.save_root_dir+"/args.txt", mode="w") as f:
            json.dump(args.__dict__, f, indent=4)

        # set opt, loss
        self.loss = nn.BCELoss()
        self.opt = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

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
            spk = t.rsplit('/', 1)
            self.writer.add_scalars(spk[0], {spk[1]: e}, self.global_step)

        # SCORE
        for t, s in self.score_dict.items():
            self.writer.add_scalar(t, s, self.global_step)

    def save_weights(self, name_head):
        # -- Save Weights of Model --
        torch.save({'epoch': self.epoch + 1, 'state_dict': self.model.state_dict()},
                   '%s/%s_ep%04d_%s.pth' % (self.weight_dir, name_head, self.epoch, self.args.model))

    def test(self):
        self.model.eval()

        errs = []
        predicts = []
        gts = []
        
        with torch.no_grad():
            # Load model weight from path of args.load_weights
            if self.args.load_weights != " ":
                path = self.args.load_weights
                pretrained_dict = torch.load(path)['state_dict']

                try:
                    self.model.load_state_dict(pretrained_dict)
                except IOError:
                    raise IOError("Xception Model Weights not found")
                print("Loaded weights.")

            pbar = tqdm(self.dataloader['test'], leave=True, ncols=100, 
                                            total=len(self.dataloader['test']))
            for i, data in enumerate(pbar):

                input_, real_, gt_, lb_ = (d.to('cuda') for d in data)
                predict_ = self.model(input_)
                t_pre_ = threshold(predict_)
                m_pre_ = morphology_proc(t_pre_)

                gts.append(gt_.permute(0,2,3,4,1).cpu().numpy())
                predicts.append(predict_.permute(0,2,3,4,1).cpu().numpy())

                errs.append(self.loss(predict_, gt_).item())

                self.color_video_dict.update({
                        'test/input-real': torch.cat([input_, real_], dim=3),
                    })
                self.gray_video_dict.update({
                        'test/mask-pre-th-mor': torch.cat([gt_, predict_, t_pre_, m_pre_], dim=3)
                    })

                pbar.set_description("[TEST Epoch %d/%d]" % (self.epoch, self.args.ep))

            gts = np.asarray(np.stack(gts), dtype=np.int32).flatten()
            predicts = np.asarray(np.stack(predicts)).flatten()
            roc = evaluate(gts, predicts, self.best_roc, self.epoch, self.save_root_dir, metric='roc')
            pr = evaluate(gts, predicts, self.best_pr, self.epoch, self.save_root_dir, metric='pr')
            f1 = evaluate(gts, predicts, metric='f1_score')

            if roc > self.best_roc:
                self.best_roc = roc
                self.save_weights('ROC')
            elif pr > self.best_pr:
                self.best_pr = pr
                self.save_weights('PR')

            self.errors_dict.update({
                    'loss/err/test': np.mean(errs)
                })
            self.score_dict.update({
                    "score/roc": roc,
                    "score/pr": pr,
                    "score/f1": f1,
                })

    def train(self):

        for self.epoch in range(self.args.ep):
            self.model.train()
            pbar  = tqdm(self.dataloader['train'], leave=True, ncols=100,
                                    total=len(self.dataloader['train']))

            for i, data in enumerate(pbar):
                
                self.global_step += 1
 
                input, real, gt, lb = (d.to('cuda') for d in data)
                self.opt.zero_grad()
                predict = self.model(input)  # Training
                err = self.loss(predict, gt) # Calc loss
                err.backward() # Backward
                self.opt.step()
                
                t_pre = threshold(predict)
                m_pre = morphology_proc(t_pre)

                self.color_video_dict.update({
                            "train/input-real": torch.cat([input, real], dim=3)
                        })
                self.gray_video_dict.update({
                            "train/gt-pre-th-mor": torch.cat([gt, predict, t_pre, m_pre], dim=3)
                    })
                self.errors_dict.update({
                        'loss/err/train': err.item()
                    })

                if self.global_step % self.args.test_freq == 0:
                    self.test()
                if self.global_step % self.args.display_freq == 0:
                    self.update_summary()
                    
                pbar.set_description("[TRAIN Epoch %d/%d]" % (self.epoch+1, self.args.ep))

        print("Training model Done.")
        

