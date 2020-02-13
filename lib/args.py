
import torch
import argparse

class Args():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--gpu', default='0', type=str, help='GPU number. Default=0')
        self.parser.add_argument('--ep', default=10, type=int, help='epochs for training. Default=10')

        # Path
        TR_PLIST = "/mnt/fs2/2018/ohshiro/vfd/lists/semi_med_train_path_list.txt"
        TS_PLIST = "/mnt/fs2/2018/ohshiro/vfd/lists/semi_med_test_path_list.txt"
        RESULT_PATH = "/mnt/fs2/2018/ohshiro/research/vfd/results"
        self.parser.add_argument('--tr_plist', default=TR_PLIST, type=str, help='train data path list. ')
        self.parser.add_argument('--ts_plist', default=TS_PLIST, type=str, help='test data path list. ')
        self.parser.add_argument('--result_root', default=RESULT_PATH, type=str, help='save any result path.')

        # Dataloader
        self.parser.add_argument('--isize', default=128, type=int, help='input frame size. Default=128')
        self.parser.add_argument('--ich', default=3, type=int, help='input channel size, RGB=3. Default=3')
        self.parser.add_argument('--nfr', default=16, type=int, help='input num frame. Default=16')
        self.parser.add_argument('--batchsize', default=4, type=int, help='input batch size. Default=16')
        self.parser.add_argument('--workers', default=4, type=int, help='num_workers. Default=4')

        # Network
        self.parser.add_argument('--model', default="mygan", type=str, help='train model. MyGANmodel=mygan, MySpatialTempmodel=c2plus1d, ConvLSTM=clstm, XceptionNet=xception. Default=mygan ')

        # Train
        self.parser.add_argument('--lr', default=2e-5, type=float, help='initial learning rate for adam. Default=2e-5')
        self.parser.add_argument('--beta1', default=0.5, type=float, help='momentum term of adam. Default=0.5')
        self.parser.add_argument('--w_adv', default=1, type=int, help='adversarial loss weight. Default=1')
        self.parser.add_argument('--w_con', default=10, type=int, help='adversarial loss weight. Default=10')
        self.parser.add_argument('--pos_weight', default=2, type=int, help='weighted BCE parameter. Default=2')
        self.parser.add_argument('--freq', default=50, type=int, 
                                help='frequency of update tensorboard and test. Default=50')

        self.parser.add_argument('--resume', default="", type=str, help='Pretrained Model weight path for training')
        self.parser.add_argument('--ae', default=False, action="store_true", help='Use AutoEncoder on c2plus1d net as Generator')

    def parse(self):
        
        self.args = self.parser.parse_args()

        str_ids = self.args.gpu.split(',')
        self.args.gpu = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0 :
                self.args.gpu.append(id)

        torch.cuda.set_device(self.args.gpu[0])

        return self.args
