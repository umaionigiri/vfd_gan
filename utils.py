
import cv2
from PIL import Image
import numpy as np

import torch
import torchvision.transforms 
from videotransforms import video_transforms, volume_transforms


def tensor_to_cv2(video):
    # B, C, D, H, W
    cv2_videos = []
    for v in video.transpose(1, 2):
        cv2_imgs = [np.asarray(torchvision.transforms.ToPILImage()(img)) for img in v]
        cv2_videos.append(np.stack(cv2_imgs))
    return np.stack(cv2_videos)


def video_to_flow(video):
    video = video.cpu()
    # tensor to pil
    # video = (B, C, D, H, W)
    cv2_video = tensor_to_cv2(video)
    # cv2_video = (B, D, C, H, W)
    
    transform = video_transforms.Compose([
            volume_transforms.ClipToTensor()
        ])

    
    flow_videos = []
    for v in cv2_video:
        flow_imgs = []
        for i, img in enumerate(v):
            next_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if i == 0:
                hsv_mask = np.zeros_like(img)
                hsv_mask[:,:,1] = 255
            else:
                flow = cv2.calcOpticalFlowFarneback(prv_img, next_img, 
                                                    None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1], angleInDegrees=True)
                hsv_mask[:,:,0] = ang / 2
                hsv_mask[:,:,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

                bgr = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2RGB)
                cv2.imwrite("/mnt/fs2/2018/ohshiro/opt_flow.png", bgr)
                bgr = Image.fromarray(np.uint8(bgr))
                flow_imgs.append(bgr)
            prv_img = next_img
        flow_imgs.append(bgr)
        flow_imgs = transform(flow_imgs)
        flow_videos.append(flow_imgs)
        
    return torch.stack(flow_videos)
 

