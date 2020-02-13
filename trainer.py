
from __future__ import print_function
import os

from lib.args import Args

def main(args):
    

    # -- DATA LOAD --
    from lib.data import DataLoader
    dataloader = DataLoader(args)
    dataloader = dataloader.load_data()

    # -- MODEL LOAD --
    print("--Load model--")
    # proposed model
    if args.model == 'mygan':
        from models.mygannet import MyGAN
        model = MyGAN(args, dataloader)
    # comparision model
    elif args.model == 'anogan':
        from models.anogan import AnoGAN
        model = AnoGAN(args, dataloader)
    #elif args.model == 'ganomaly':
        #from ganomaly import Ganomaly
        #model = Ganomaly(args, dataloader)

    elif args.model == 'c2plus1d':
        from lib.train_stcnn import VFD_STCNN
        model = VFD_STCNN(args, dataloader)
    elif args.model == 'xception':
        from lib.train_stcnn import VFD_STCNN
        model = VFD_STCNN(args, dataloader)
    elif args.model == 'clstm':
        from lib.train_stcnn import VFD_STCNN
        model = VFD_STCNN(args, dataloader)
    else:
        print("\n %s is None." % (args.model))
        exit()

    model.train()
        

if __name__ == '__main__':

    args = Args().parse()

    print(args.gpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if len(args.gpu) >1 :
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpu)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu[0])

    main(args)


