import cv2
import torch
from PIL import Image
import glob
import os 
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image

from videotransforms import video_transforms, volume_transforms


class MdfDataLoader(Dataset):
    def __init__(self, isize, nfr, path_li, transforms=None):
        # set self
        self.isize = isize
        self.nfr = nfr
        self.paths = path_li
        self.transforms = transforms
        self.mask_transforms = video_transforms.Compose([
                                video_transforms.Resize((self.isize, self.isize)),
                                volume_transforms.ClipToTensor(channel_nb=1)
                                ])
        
        # Set index
        self.data_path_li, self.lb_path_li, self.mask_path_li= self.path_reader(self.paths) #video path list
        nframe_li = self.count_frame(self.mask_path_li) #num of frame list
        div_nfr_li = [ i // self.nfr for i in nframe_li] #num of nfrsize list
        # div_nfr_li -> data index
        self.total_div_nfr = div_nfr_li
        for i in range(len(div_nfr_li)):
            if i != 0: self.total_div_nfr[i] += self.total_div_nfr[i-1]
        
    def path_reader(self, path_list):
        data_path = [line.rstrip() for line in open(path_list)]
        mask_path = []
        lb_path = []
        for video in data_path:
            root = video.rsplit("/", 1)[:-1]
            name = root[0].rsplit("/", 1)[-1]
            mask_path.append( os.path.join(root[0], "[Mask]" + name + ".mp4") )
            lb_path.append( os.path.join(root[0], "[Original]" + name + ".mp4") )
        return data_path, lb_path, mask_path

    def count_frame(self, path):
        nframe_li = []
        for i, p in enumerate(path):
            cap = cv2.VideoCapture(p)
            nframe_li.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            cap.release()
        return nframe_li
           
    def __getitem__(self, index):
        # get start point
        video_id, ff = self.get_first_frame(index)
        # read video data
        frsize_data = self.video_reader(self.data_path_li[video_id], ff)

        if "Fake" in self.data_path_li[video_id]:
            frsize_lb = self.video_reader(self.lb_path_li[video_id], ff)
            frsize_mask = self.video_reader(self.mask_path_li[video_id], ff, mask=True)
            transdata = frsize_data + frsize_lb + frsize_mask
            if self.transforms: 
                transdata = self.transforms(transdata)
            frsize_data, frsize_lb, frsize_mask = torch.split(transdata, self.nfr, dim=1)

        elif "Original" in self.data_path_li[video_id]:
            frsize_mask = torch.zeros((1, self.nfr, self.isize, self.isize))
            if self.transforms:
                frsize_data = self.transforms(frsize_data)
            frsize_lb = frsize_data
        
        return frsize_data*2-1, frsize_lb*2-1, frsize_mask[0].unsqueeze(0)

    def __len__(self):
        return self.total_div_nfr[-1]
 
    def get_first_frame(self, index):
        for i, v in enumerate(self.total_div_nfr):
            if v >= index:
                if i==0: first_frame = (index-1) * self.nfr
                else: first_frame = (index-self.total_div_nfr[i-1] - 1 ) * self.nfr
                return i, first_frame
    

    def video_reader(self, video_path, ff, mask=False):
        
        data = []
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, ff)
        
        for i in range(self.nfr):
            ret, frame = cap.read()
            if mask == True:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.bitwise_not(frame)
                frame = cv2.merge([frame, frame, frame])
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame)) 
            data.append(frame)
        cap.release()
        return data
  

class DataLoader(object):
    def __init__(self, args):
        # Initialozation
        self.args = args
        self.isize = args.isize
        self.nfr = self.args.nfr
        self.plist = {'train': args.tr_plist, 'test': args.ts_plist}

        # set transforms
        train_transforms = video_transforms.Compose([
                            video_transforms.Resize((int(self.isize*1.1),int(self.isize*1.1))),
                            video_transforms.RandomRotation(10),
                            video_transforms.RandomCrop((self.args.isize, self.args.isize)),
                            video_transforms.RandomHorizontalFlip(),
                            #video_transforms.ColorJitter(),
                            video_transforms.Resize((self.isize, self.isize)),
                            volume_transforms.ClipToTensor()
                            ])
        test_transforms = video_transforms.Compose([
                            video_transforms.Resize((self.isize, self.isize)),
                            volume_transforms.ClipToTensor()
                            ])
        self.transforms = {'train': train_transforms, 'test':test_transforms}
    
    def load_data(self):
        
        print("load Data")
        splits = ['train', 'test']
        shuffle = {'train': True, 'test': False}

        # dataset
        dataset = {}
        loader = lambda x: MdfDataLoader(self.isize, self.nfr, 
                                        self.plist[x], transforms=self.transforms[x])
        dataset['train'] = loader('train')
        dataset['test'] = loader('test')
       
        # dataloader
        dataloader = { x: torch.utils.data.DataLoader(  
                                dataset=dataset[x],
                                batch_size = self.args.batchsize,
                                drop_last=True,
                                shuffle=shuffle[x],
                                num_workers=self.args.workers
                                )
                        for x in splits }
        return dataloader
            

