

from __future__ import print_function
import os 
import numpy as np


def Args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_netg_path', type=str, help='NetG weight path')
    parser.add_argument('--test_path', type=str, help='test data path')

    
    return pargser.pargse_args()

def video_reader(path, mask=False):
    data = []
    cap = cv2.VideoCapture(path)
    
    while(True):
        ret, frame = cap.read()
        if mask == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.bitwise_not(frame)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frme = Image.fromarray(np.uint8(frame))
        data.append(frame)
        cap.release()
        return data

def load_data(test_path):
    mask_path = []
    data_path = [line.rstrip() for line in open(test_path)]
    for dp in data_path:
        root = p.rsplit("/", 1)[:-1]
        name = root[0].rsplit("/", 1)[-1]
        mask_path = os.path.join(root[0], "[Mask]" + name + ".mp4")
    
    for dp, mp in zip(data_path, mask_path):
        data = video_reader(dp)
        mask = video_reader(mp, mask=True)


def main():

    #args
    args = Args()
    
    #load weight
    pretrained_dict = torch.load(args.weight_netg_path)['state_dict']
    try:
        netg.load_state_dict(pretrained_dict)
    except IOError:
        raise IOError("netG weights not found")
    print("Loaded weights")

    #load data
    



    #test



if __name__ == '__main__':
    main()
