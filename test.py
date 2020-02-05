

import argparse
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset 
from torchvision import transforms 
from torchvision.utils import save_image

from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score, precision_recall_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import rc

from data import MdfDataLoader
from utils import *
from videotransforms import video_transforms, volume_transforms

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--isize', type=int, default=128)
parser.add_argument('--nfr', type=int, default=16)
parser.add_argument('--batchsize', type=int, default=4)
parser.add_argument('--metric', type=str, default='roc')
parser.add_argument('--test_data_path', type=str, default='')
parser.add_argument('--test_model_list_path', type=str, default='')
args = parser.parse_args()

SAVEROOT = "/mnt/fs2/2018/ohshiro/vfd/results/test/"


#GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
   
fig=plt.figure()

def evaluate(labels, scores, metric=None):
    if metric == 'roc':
        return roc(labels, scores)
    elif metric == 'auprc':
        return auprc(labels, scores)
    elif metric == 'pr':
        return pr(labels, scores, best, iter, saveto)
    elif metric == 'f1_score':
        threshold = 0.20
        scores[scores >= threshold] = 1
        scores[scores <  threshold] = 0
        return f1_score(labels, scores)
    else:
        raise NotImplementedError("Check the evaluation metric.")

##
def roc(labels, scores, name):
    """Compute ROC curve and ROC area for each class"""
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Equal Error Rate
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    lw = 2
    plt.plot(fpr, tpr, lw=lw, label='%s: (AUC = %0.2f, EER = %0.2f)' % (name, roc_auc, eer))
    plt.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
    plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    
    return roc_auc

def auprc(labels, scores):
    ap = average_precision_score(labels, scores)
    return ap


def pr(labels, scores, name):
    precision, recall, th = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)

    lw = 2
    plt.plot(recall, precision, label='%s: (AUC = %0.2f)' % (name, pr_auc))
    plt.plot([0,1], [1,0], color='navy', lw=1, linestyle=':')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower right')

    return pr_auc


def load_model(m):
    # LOAD MODEL
    if 'ganbase' in m:
        from network import NetG
        model = NetG()
        name = "ganbase"
    elif 'c2plus1d' in m:
        from vfd_c2plus1d import AutoEncoder
        model = AutoEncoder()
        name = "c2plus1d"
    elif 'xception' in m:
        from xception import Xception
        model = Xception()
        name = "xception"
    elif 'clstm' in m:
        from convlstm import ConvLSTMModel
        model = ConvLSTMModel(args)
        name = "clstm"
    else:
        print("Weight path not found.")
        exit()
    model = model.cuda()
    # Apply Weight
    trained_dict = torch.load(m, map_location='cuda:0')['state_dict']
    try:
        model.load_state_dict(trained_dict)
    except IOError:
        raise IOError("MODEL weights not fount")
        
    return model, name

def test():
    step = 0

    # Data load
    transforms = video_transforms.Compose([
            video_transforms.Resize((args.isize, args.isize)),
            volume_transforms.ClipToTensor()
        ])
    dataset = MdfDataLoader(args.isize, args.nfr, args.test_data_path, transforms)
    dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size = args.batchsize,
                drop_last = True,
                shuffle=False,
                num_workers=8
            )
    
    model_list = [line.rstrip() for line in open(args.test_model_list_path)]

    with torch.no_grad():
        for m_i, m in enumerate(model_list):
            print(m)
            model, name = load_model(m)
            
            save_root = os.path.join(SAVEROOT, "iamges", name)
            if not os.path.exists(save_root): os.makedirs(save_root)
            print(save_root)
 
            gts = []
            predicts = []
            pbar = tqdm(dataloader, leave=True, ncols=100, total=len(dataloader))

            for i, data in enumerate(pbar):

                input, real, gt, lb = (d.to('cuda') for d in data)
                predict = model(input)
                t_pre = threshold(predict)
                m_pre = morphology_proc(t_pre)

                gts.append(gt.permute(0,2,3,4,1).cpu().numpy())
                predicts.append(predict.permute(0,2,3,4,1).cpu().numpy())
                
                # -- SAVE IMAGE --
                grid = torch.cat([normalize(input), normalize(real), gray2rgb(gt), gray2rgb(predict), gray2rgb(t_pre), gray2rgb(m_pre)], dim=3)
                for image in grid.permute(0,2,1,3,4):
                    save_image(image, os.path.join(save_root, "%06d.png" % (step)), nrow=args.nfr)
                    step += 1

                pbar.set_description("[TEST %d/%d]" % (m_i+1, len(model_list)))
           
           
            gts = np.asarray(np.stack(gts), dtype=np.int32).flatten()
            predicts = np.asarray(np.stack(predicts)).flatten()
            if args.metric == 'roc':
               score = roc(gts, predicts, name)
            elif args.metric == 'pr':            
               score = pr(gts, predicts, name)
            f1 = evaluate(gts, predicts, metric='f1_score')
            print("%s / %s == %f" % (m, args.metric, score))
            print("%s / f1 == %f" % (m, f1))
        plt.savefig(os.path.join(SAVEROOT, "%s_curve.png" % (args.metric)))

if __name__ == '__main__':

    test()



