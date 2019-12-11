""" 
Creates an Xception Model as defined in:

Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf

This weights ported from the Keras implementation. Achieves the following performance on the validation set:

Loss:0.9173 Prec@1:78.892 Prec@5:94.292

REMEMBER to set your image size to 3x299x299 for both test and validation

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])

The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch

__all__ = ['xception']

model_urls = {
    'xception':'https://www.dropbox.com/s/1hplpzet9d7dv29/xception-c0a72b38.pth.tar?dl=1'
}


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,ksize=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv3d(in_channels,in_channels,(ksize, ksize, 1),stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv3d(in_channels,out_channels,(1, 1, ksize),1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv3d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm3d(out_filters)
        else:
            self.skip=None
        
        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm3d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm3d(filters))
        
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm3d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool3d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x



class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, args, num_classes=1000):
        super(Xception, self).__init__()

        self.args = args
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv3d(self.args.ich, 32, (3, 3, 3), stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm3d(64)
        #do relu hered

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm3d(1536)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm3d(2048)

        #self.fc = nn.Linear(2048, num_classes)
        
        #decode
        #4 -> 8
        self.convt1 = nn.ConvTranspose3d(2048, 1024, 4, stride=2, padding=1) 
        self.bn5 = nn.BatchNorm3d(1024)
        #8 -> 16
        self.convt2 = nn.ConvTranspose3d(1024, 512, 4, stride=2, padding=1)
        self.bn6 = nn.BatchNorm3d(512)
        #16 -> 32
        self.convt3 = nn.ConvTranspose3d(512, 256, 4, stride=2, padding=1)
        self.bn7 = nn.BatchNorm3d(256)
        #32 -> 64
        self.convt4 = nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1)
        self.bn8 = nn.BatchNorm3d(128)
        #64 -> 128
        self.convt5 = nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1)
        self.bn9 = nn.BatchNorm3d(64)
        self.conv5 = nn.Conv3d(64, 1, 3, padding=(0, 1, 1))
        self.sigmoid = nn.Sigmoid()


        #------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #-----------------------------





    def forward(self, x):
        x = x.transpose(1,2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.convt1(x)
        x = self.bn5(x)
        x = self.convt2(x)
        x = self.bn6(x)
        x = self.convt3(x)
        x = self.bn7(x)
        x = self.convt4(x)
        x = self.convt5(x)
        
        x = self.conv5(x)
        x = self.sigmoid(x)

        return x

def train(args, dataloader):
    print("Xception Model")
    model = Xception(args)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
    writer = SummaryWriter()
    n_iter = 0
    model.train()

    model.cuda()
    
    for epoch in range(args.ep):
        with tqdm(dataloader['train'], ncols=100) as pbar:
            for i, data in enumerate(pbar):
                data, target = (d.to('cuda') for d in data)
                print(data.shape)
                optimizer.zero_grad()
            
                output = model(data)
                target = target.transpose(1,2)
                loss = loss_fn(output, target)

                #tqdm
                pbar.set_description("[Epoch %d]" % (epoch+1))
                pbar.set_postfix(OrderedDict(loss="{:.4f}".format(loss)))

                #tensorboard

                grid = torchvision.utils.make_grid(data.view(-1, 3, 128, 128), \
                        nrow=int(np.log2(args.batchsize*args.nfr)), normalize=True)
                out_grid = torchvision.utils.make_grid(output.view(-1, 1, 128, 128), \
                        nrow=int(np.log2(args.batchsize*args.nfr)), normalize=True)
                tar_grid = torchvision.utils.make_grid(target.view(-1, 1, 128, 128), \
                        nrow=int(np.log2(args.batchsize*args.nfr)), normalize=True)

                #writer.add_image('input_image', grid, n_iter)
                #writer.add_image('output_image', out_grid, n_iter)
                #writer.add_image('target image', tar_grid, n_iter)
                writer.add_video('input', data, n_iter)
                writer.add_video('target', target, n_iter)
                writer.add_video('output', output, n_iter)
            
                writer.add_scalar('Loss/train', loss, n_iter)

                n_iter += 1
                loss.backward()
                optimizer.step()

        save_path = os.path.join(save_dir, args.trdataset, \
                (args.trdataset+'_e{:04d}.pt').format(epoch))
        #torch.save(state_dict, save_path)
        model.eval()




def xception(args, pretrained=False):
    """
    Construct Xception.
    """

    model = Xception(args)
    """
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['xception']))
    """
    return model



