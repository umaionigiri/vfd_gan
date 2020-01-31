""" Evaluate ROC

Returns:
    auc, eer: Area under the curve, Equal Error Rate
"""

# pylint: disable=C0103,C0301

##
# LIBRARIES
from __future__ import print_function

import csv
import os
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score, precision_recall_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import rc
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#rc('text', usetex=True)
fig=plt.figure()

def evaluate(labels, scores, best=None, iter=None, saveto=None, metric=None):
    if metric == 'roc':
        return roc(labels, scores, best, iter, saveto)
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
def roc(labels, scores, best, iter, saveto=None):
    """Compute ROC curve and ROC area for each class"""
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    #labels = labels.cpu()
    #scores = scores.cpu()

    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Equal Error Rate
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    if roc_auc > best:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f, EER = %0.2f)' % (roc_auc, eer))
        plt.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
        plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(saveto, "ROC_%03d.png" % (iter)))
        plt.close()
        
        with open(os.path.join(saveto,'roc_%03d' % (iter)), 'w', newline='') as f:
            writer = csv.writer(f)
            for data in zip(fpr, tpr):
                writer.writerow(data)

    return roc_auc

def auprc(labels, scores):
    ap = average_precision_score(labels, scores)
    return ap


def pr(labels, scores, best, iter, saveto=None):
    precision, recall, th = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)
    
    if pr_auc > best:
        plt.figure()
        lw = 2
        plt.plot(recall, precision, label='(AUC = %0.2f)' % (pr_auc))
        plt.plot([0,1], [1,0], color='navy', lw=1, linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(saveto, 'PR_%03d.png' % (iter)))
        plt.close()

        with open(os.path.join(saveto, 'pr_%03d' % (iter)), 'w', newline='') as f:
            writer = csv.writer(f)
            for data in zip(recall, precision):
                writer.writerow(data)
    
    return pr_auc



