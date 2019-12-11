import cv2
from torchvision.transforms import ToPILImage, ToTensor
import torch
import torch.nn as nn

def weights_init(m):
    if isinstance(m, nn.Conv3d):
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, nn.BatchNorm3d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    

class NetgConv(nn.Module):
    def __init__(self, in_fi, out_fi):
        super(NetgConv, self).__init__()

        self.conv = nn.Conv3d(in_fi, out_fi, 3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(out_fi)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)
        return x


class NetG(nn.Module):
    def __init__(self, nc=3, ngf=32):
        super(NetG, self).__init__()

        self.dconv1 = NetgConv(nc, ngf)
        self.dconv2 = NetgConv(ngf, ngf*2)
        self.dconv3 = NetgConv(ngf*2, ngf*4)
        self.dconv4 = NetgConv(ngf*4, ngf*8)
        self.dconv5 = NetgConv(ngf*8, ngf*16)

        self.maxpool = nn.MaxPool3d(2)
        
        self.uconv5 = NetgConv(ngf*16, ngf*8)
        self.uconv4 = NetgConv(ngf*8+ngf*8, ngf*8)
        self.uconv3 = NetgConv(ngf*8+ngf*4, ngf*4)
        self.uconv2 = NetgConv(ngf*4+ngf*2, ngf*2)
        self.uconv1 = NetgConv(ngf*2+ngf, ngf)
        
        self.dropout = nn.Dropout(p=0.25)
        self.upsamp = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.conv_last = nn.Conv3d(ngf, nc, 3, stride=1, padding=1, bias=False)

    def forward(self, x):

        # Encode 1
        # (32, 128)
        dconv1 = self.dconv1(x) # ngf
        x = self.maxpool(dconv1)
        # (16, 64)
        dconv2 = self.dconv2(x) # ngf*2
        x = self.maxpool(dconv2)
        # (8, 32)
        dconv3 = self.dconv3(x) # ngf*4
        x = self.maxpool(dconv3)
        # (4, 16)
        dconv4 = self.dconv4(x) # ngf*8
        x = self.maxpool(dconv4)
        # (2, 8)

        latent_i = self.dconv5(x) # ngf*16

        #Decoder
        x = self.uconv5(latent_i) # ngf*8
        x = self.dropout(x)
        x = self.upsamp(x) # ngf*8
        # (4, 8)
        x = torch.cat([x, dconv4], dim=1) # ngf*8*2
        x = self.uconv4(x) # ngf*8
        x = self.dropout(x)
        x = self.upsamp(x)
        # (4, 16)
        x = torch.cat([x, dconv3], dim=1) #ngf*8+ngf*4
        x = self.uconv3(x)
        x = self.dropout(x)
        x = self.upsamp(x)
        # (8, 32)
        x = torch.cat([x, dconv2], dim=1)
        x = self.uconv2(x)
        x = self.dropout(x)
        x = self.upsamp(x)
        # (16, 64)
        x = torch.cat([x, dconv1], dim=1)
        x = self.uconv1(x)

        gen_vi = self.conv_last(x)
        
        """
        # Encode 2
        # (32, 128) -> (16, 64)
        dconv1 = self.dconv1(gen_vi)
        x = self.maxpool(dconv1)
        # (16, 64) -> (8, 32)
        dconv2 = self.dconv2(x)
        x = self.maxpool(dconv2)
        # (8, 32) -> (4, 16)
        dconv3 = self.dconv3(x)
        x = self.maxpool(dconv3)
        # (4, 16) -> (2, 8)
        dconv4 = self.dconv4(x)
        x = self.maxpool(dconv4)

        latent_o = self.dconv5(x)
        """


        return latent_i, gen_vi
        

class NetdConv(nn.Module):
    def __init__(self, in_fi, out_fi, kernel=None, padding=None):
        super(NetdConv, self).__init__()

        self.conv = nn.Conv3d(in_fi, out_fi, kernel, stride=1, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_fi)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)
        return x


class Disc(nn.Module):
    def __init__(self, nc, nfr, ndf=32, kernel=None, padding=None):
        super(Disc, self).__init__()

        # input size == (B, C, D, H, W)
        netgconv = lambda in_fi, out_fi: NetdConv(in_fi, out_fi, kernel=kernel, padding=padding)
        self.dconv1 = netgconv(nc, ndf)
        self.dconv2 = netgconv(ndf, ndf*2)
        self.dconv3 = netgconv(ndf*2, ndf*4)
        self.dconv4 = netgconv(ndf*4, ndf*8)
        self.dconv5 = netgconv(ndf*8, ndf*16)
        self.dconv6 = netgconv(ndf*16, ndf*32)

        self.maxpool = nn.MaxPool3d((1, 2, 2)) 
        self.gpool = nn.AvgPool3d((nfr, 1, 1), stride=1)
        self.linear = nn.Linear(ndf*32*2*2, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # (32, 128)
        x = self.dconv1(x)
        x = self.maxpool(x)
        # (32, 64)
        x = self.dconv2(x)
        x = self.maxpool(x)
        # (32, 32)
        x = self.dconv3(x)
        x = self.maxpool(x)
        # (32, 16)
        x = self.dconv4(x)
        x = self.maxpool(x)
        # (32, 8)
        x = self.dconv5(x)
        x = self.maxpool(x)
        # (32, 4)
        x = self.dconv6(x)
        features = self.maxpool(x)
        # (32, 2)
        x = self.gpool(features) # nfr -> 1
        x = self.linear(x.view(x.shape[0], -1))
        classifier = self.sigmoid(x)

        return classifier.squeeze(1), features


class NetD(nn.Module):
    def __init__(self, args):
        super(NetD, self).__init__()

        self.spatdisc = Disc(args.ich, args.nfr, kernel=(1, 3, 3), padding=(0, 1, 1))
        self.tempdisc = Disc(args.ich, args.nfr, kernel=(1, 3, 3), padding=(0, 1, 1))

    def forward(self, x, y):

        s_cls, s_feat = self.spatdisc(x)
        t_cls, t_feat = self.tempdisc(y)

        return s_cls, s_feat, t_cls, t_feat
        

