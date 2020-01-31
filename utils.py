
import cv2
from PIL import Image
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms 
from videotransforms import video_transforms, volume_transforms
from torchvision.utils import save_image, make_grid


def weights_init(m):
    if isinstance(m, nn.Conv3d):
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, nn.BatchNorm3d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    

def l2_loss(inp, target, size_average=True):
    if size_average:
        return torch.mean(torch.pow((inp-target), 2))
    else:
        return torch.pow((inp-target), 2)

def bce_smooth(inputs, target, weight=None, reduction='mean', smooth_eps=None, from_logits=False):
    smooth_eps = smooth_eps or 0
    if smooth_eps > 0:
        target = target.float()
        target.add_(smooth_eps).div_(2.)
    return F.binary_cross_entropy(inputs, target, weight=weight, reduction=reduction)


def calc_pos_wights(output):
    c_zero = torch.sum(output == 0)
    c_non_zero = torch.sum(output > 0)
    print("{} , {} ".format(c_zero, c_non_zero))
    weights = torch.tensor(c_zero/c_non_zero)
    return weights

def normalize(tensor):
    """
    shift tensor image to the range (0, 1)
    """
    tensor = tensor.clone()
    min = float(tensor.min())
    max = float(tensor.max())
    tensor.clamp_(min=min, max=max)
    return tensor.add_(-min).div_(max-min+1e-5)

def gray2rgb(video):
    return torch.cat([video, video, video], dim=1)

def video_to_flow(video):
    # video tensor(-1, 1) to ndarray(0, 255)
    norm_video = [normalize(v) for v in video.permute(2, 0, 1, 3, 4)] # (D, B, C, W, H)
    norm_video = torch.stack(norm_video).permute(1, 0, 3, 4, 2) # (B, D, W, H, C)
    nd_video = norm_video.cpu().numpy()
    
    transform = video_transforms.Compose([
            volume_transforms.ClipToTensor()
        ])
    # calc optical flow 
    flow_videos = []
    for v in nd_video:
        flow_imgs = []
        for i, img in enumerate(v):
            next_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            #next_img = img
            if i == 0:
                hsv_mask = np.zeros_like(img)
                hsv_mask[:,:,1] = 255
            else:
                flow = cv2.calcOpticalFlowFarneback(prv_img, next_img, 
                                                    None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1], angleInDegrees=True)
                hsv_mask[:,:,0] = ang / 2
                hsv_mask[:,:,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

                rgb = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2RGB)
                #cv2.imwrite("/mnt/fs2/2018/ohshiro/opt_flow_.png", rgb)
                rgb = Image.fromarray(np.uint8(rgb))
                flow_imgs.append(rgb)
            prv_img = next_img
        flow_imgs.append(rgb)
        flow_imgs = transform(flow_imgs)
        flow_videos.append(flow_imgs)
        
    return torch.stack(flow_videos)*2-1
 
def rgb_to_gray(video):
    gray_video = []
    for v in video:
        gray_img = [cv2.cvtColor(i, cv2.COLOR_RGB2GRAY) for i in v]
        gray_video.append(np.stack(gray_img))
    return np.expand_dims(np.stack(gray_video), axis=-1)


def morphology_proc(video):
    morph_video = []
    video = video.cpu().numpy()
    kernel = np.ones((5, 5), np.uint8)
    for v in video:
        op_img = [cv2.morphologyEx(i, cv2.MORPH_OPEN, kernel) for i in v]
        #cl_img = [cv2.morphologyEx(i, cv2.MORPH_OPEN, kernel) for i in v]
        morph_video.append(np.stack(op_img))
    return torch.from_numpy(np.stack(morph_video)).cuda()

def threshold(data):
    t = torch.Tensor([0.45]).to('cuda')
    out = (data > t).float() * 1
    return out

def predict_forg(gout, input):
    # predict image
    # diff video
    diff = torch.abs(gout - input)
    # normalize diff -> (0, 1)
    diff_norm = [normalize(v) for v in diff.permute(2, 0, 1, 3, 4)]
    diff_norm = torch.stack(diff_norm).permute(1, 0, 3, 4, 2) # (B, D, W, H, C)
    # tensor to numpy
    diff_norm = diff_norm.cpu().numpy()
    # post processing 
    diff_norm = rgb_to_gray(diff_norm)
    predict = diff_norm
    #predict = np.expand_dims(morphology_proc(predict), axis=1)
    return torch.from_numpy(predict).permute(0, 4, 1, 2, 3)
