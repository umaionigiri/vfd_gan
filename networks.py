import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel

def weights_init(mod):
    if isinstance(mod, nn.Conv3d):
        n = mod.kernel_size[0] * mod.kernel_size[1] * mod.out_channels
        mod.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(mod, nn.BatchNorm3d):
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv3d(in_channels, in_channels, (1, ksize, ksize), stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv3d(in_channels, out_channels, (ksize, 1, 1), stride=1, padding=0, dilation=1, groups=1, bias=bias)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x



class Encoder(nn.Module):
    """
    DCGAN Encoder Network
    ngf : output feature size in each convolution

    """
    def __init__(self, isize, nfr, nz, nc, ngf, n_extra_layers=0, add_final_conv=True):
        super(Encoder, self).__init__()

        assert isize % 16 == 0, "isize has to be a multiple of 16"

        #downconv
        #(32, 128, 128) -> (16, 64, 64)
        self.conv1_1 = nn.Conv3d(nc, 32, 4, stride=2, padding=1, bias=False)
        self.conv1_2 = nn.Conv3d(32, 32, 3, stride=1, padding=1, bias=False)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        #self.pool1 = nn.MaxPool3d(3, stride=2, padding=1)
        #conv
        #(16, 64, 64) -> (8, 16, 16)
        self.conv2_1= nn.Conv3d(32, 64, 4, stride=2, padding=1, bias=False)
        self.conv2_2 = nn.Conv3d(64, 64, (3, 4, 4), stride=(1, 2, 2), padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(64)
        #self.pool2 = nn.MaxPool3d(3, stride=2, padding=1)
        
        #(8, 16, 16) -> (4, 8, 8)
        self.conv3_1 = nn.Conv3d(64, 128, 4, stride=2, padding=1, bias=False)
        self.conv3_2 = nn.Conv3d(128, 128, 3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(128)
        #self.pool3 = nn.MaxPool3d(3, stride=2, padding=1)

        #(4, 8, 8) -> (2, 2, 2)
        self.conv4_1 = nn.Conv3d(128, 256, 4, stride=2, padding=1, bias=False)
        self.conv4_2= nn.Conv3d(256, 256, (3, 4, 4), stride=(1, 2, 2), padding=1, bias=False)
        self.bn4 = nn.BatchNorm3d(256)
        #self.pool4 = nn.MaxPool3d(3, stride=2, padding=1)

        #(2, 2, 2) -> (1, 1, 1)
        self.last_layer = nn.Conv3d(256, nz, 4, stride=2, padding=1, bias=False)


    def forward(self, x):
        #print("encoder == {}".format(x.shape))
        x = self.conv1_1(x)
        #x = self.conv1_2(x)
        x = self.lrelu(x)
        #x = self.pool1(x)
        
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.bn2(x)
        x = self.lrelu(x)
        #x = self.pool2(x)

        x = self.conv3_1(x)
        #x = self.conv3_2(x)
        x = self.bn3(x)
        x = self.lrelu(x)
        #x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.bn4(x)
        x = self.lrelu(x)
        #x = self.pool4(x)

        x = self.last_layer(x)

        return x

class Decoder(nn.Module):
    """
    DCGAN Decoder Network

    """

    def __init__(self, isize, nfr, nz, nc, ngf, n_extra_layers=0):
        super(Decoder, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple if 16"
        
       # (1, 1, 1) -> (2, 2, 2)
        self.convt1 = nn.ConvTranspose3d(nz, 256, 4, stride=2, padding=1, bias=False)
        self.conv1 = nn.Conv3d(256, 256, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(256)
        self.relu = nn.ReLU(True)
        # (2, 2, 2) -> (4, 8, 8)
        self.convt2_1 = nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1, bias=False)
        self.conv2_1 = nn.Conv3d(128, 128, 3, stride=1, padding=1, bias=False)
        self.convt2_2 = nn.ConvTranspose3d(128, 128, (3, 4, 4), stride=(1, 2, 2), padding=1, bias=False)
        self.conv2_2 = nn.Conv3d(128, 128, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(128)
        # (4, 8, 8) -> (8, 16, 16)
        self.convt3 = nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv3d(64, 64, 3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(64)
        # (8, 16, 16) -> (16, 64, 64)
        self.convt4_1 = nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1, bias=False)
        self.conv4_1 = nn.Conv3d(32, 32, 3, stride=1, padding=1, bias=False)
        self.convt4_2 = nn.ConvTranspose3d(32, 32, (3, 4, 4), stride=(1, 2, 2), padding=1, bias=False)
        self.conv4_2 = nn.Conv3d(32, 32, 3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm3d(32)
        # (16, 64, 16) -> (32, 128, 128)
        self.convt5 = nn.ConvTranspose3d(32, 16, 4, stride=2, padding=1, bias=False)
        self.conv5 = nn.Conv3d(16, nc, 3, stride=1, padding=1, bias=False)
        self.tanh = nn.Tanh()



    def forward(self, x):
        
        #x = self.main(x)

        x = self.convt1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.convt2_1(x)
        #x = self.conv2_1(x)
        x = self.convt2_2(x)
        #x = self.conv2_2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.convt3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.convt4_1(x)
        #x = self.conv4_1(x)
        x = self.convt4_2(x)
        #x = self.conv4_2(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.convt5(x)
        x = self.conv5(x)
        x = self.tanh(x)

        return x


class NetD(nn.Module):
    """
    DISCRIMINATOR Network
    """
    def __init__(self, args):
        super(NetD, self).__init__()
        model = Encoder(args.isize, args.nfr, 1, args.ich, args.ngf, args.extralayers)
        layers = list(model.children())
        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        features = self.features(x)
        features = features
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)
        return classifier, features

class NetG(nn.Module):

    def __init__(self, args):
        super(NetG, self).__init__()
        self.encoder1 = Encoder(args.isize, args.nfr, args.nz, 
                            args.ich, args.ngf, args.extralayers)
        self.decoder = Decoder(args.isize, args.nfr, args.nz, 
                            args.ich, args.ngf, args.extralayers)
        self.encoder2 = Encoder(args.isize, args.nfr, args.nz, 
                            args.ich, args.ngf, args.extralayers)

    def forward(self, x):
        latent_i = self.encoder1(x)
        gen_img = self.decoder(latent_i)
        latent_o = self.encoder2(gen_img)
        return gen_img, latent_i, latent_o


