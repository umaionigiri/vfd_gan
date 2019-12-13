
from __future__ import print_function
import os

from args import Args

def main(args):
    

    #dataloader
    from data import DataLoader
    dataloader = DataLoader(args)
    dataloader = dataloader.load_data()

    #model
    if args.model == 'xcp':
        from xception import xception
        model = Xception(agrs)
        model.train(args, dataloader)
               
    elif args.model == 'ganbase':
        from ganomaly import Ganomaly
        print("--Load model--")
        model = Ganomaly(args, dataloader)
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


