import torch
import argparse

class Args():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--gpu', default='1', type=str, help='GPU number')
        self.parser.add_argument('--ep', default=100, type=int, help='epochs for training')
        self.parser.add_argument('--device', type=str, default='gpu', help='Device: gpu | cpu')

        # Path
        TR_PLIST = "/mnt/fs2/2018/ohshiro/vfd/lists/mod_mp4_train_path_list.txt"
        TS_PLIST = "/mnt/fs2/2018/ohshiro/vfd/lists/mod_mp4_test_path_list.txt"
        RESULT_PATH = "/mnt/fs2/2018/ohshiro/vfd/results"
        self.parser.add_argument('--tr_plist', default=TR_PLIST, type=str, help='train data path list ')
        self.parser.add_argument('--ts_plist', default=TS_PLIST, type=str, help='test data path list ')
        self.parser.add_argument('--result_root', default=RESULT_PATH, type=str, help='save any result path')

        # Dataloader
        self.parser.add_argument('--isize', default=128, type=int, help='input frame size')
        self.parser.add_argument('--ich', default=3, type=int, help='input channel size, RGB=3')
        self.parser.add_argument('--nfr', default=16, type=int, help='input num frame')
        self.parser.add_argument('--batchsize', default=8, type=int, help='input batch size')
        self.parser.add_argument('--workers', default=4, type=int, help='num_workers')

        # Network
        self.parser.add_argument('--model', default="ganbase", type=str, help='train model ')

        # Train
        self.parser.add_argument('--phase', default="train", type=str, help='initial learning rate for adam')
        self.parser.add_argument('--lr', default=0.0002, type=float, help='initial learning rate for adam')
        self.parser.add_argument('--beta1', default=0.5, type=float, help='momentum term of adam')
        self.parser.add_argument('--w_adv', default=1, type=float, help='adversarial loss weight')
        self.parser.add_argument('--w_con', default=50, type=float, help='adversarial loss weight')
        self.parser.add_argument('--w_pre', default=50, type=float, help='reconstruction loss weight')
        self.parser.add_argument('--display_freq', default=1, type=int, 
                                help='frequency of showing training results on tensorboard')
        self.parser.add_argument('--save_weight_freq', default=10, type=int, 
                                help='frequency of saving weights')
        self.parser.add_argument('--test_freq', default=50, type=int, 
                                help='frequency of test')


        # Test
        self.parser.add_argument('--load_weights', default=" ", type=str, help='train or test ')
        self.parser.add_argument('--metric', default="roc", type=str, help='train model ')

        self.parser.add_argument('--resume', default="", type=str, help='train model ')
        self.parser.add_argument('--isTrain', default=True, action="store_true", help='train or test ')

    def parse(self):
        
        self.args = self.parser.parse_args()

        str_ids = self.args.gpu.split(',')
        self.args.gpu = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0 :
                self.args.gpu.append(id)

        if self.args.device == 'gpu':
            torch.cuda.set_device(self.args.gpu[0])

        return self.args
