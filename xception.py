
import torch.nn as nn


__all__ = ['xception']

class SepaConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SepaConv, self).__init__()
        self.conv1 = nn.Conv3d(in_ch, in_ch, (1, 3, 3), stride=1, 
                                padding=(0, 1, 1), dilation=1, bias=False)
        self.pointwise = nn.Conv3d(in_ch, out_ch, (3, 1, 1), stride=1, 
                                padding=(1, 0, 0), dilation=1, bias=False)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class Block(nn.Module):
    def __init__(self, in_fi, out_fi, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_fi != in_fi or strides != 1:
            self.skip = nn.Conv3d(in_fi, out_fi, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm3d(out_fi)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []
        filters = in_fi

        if grow_first:
            rep.append(self.relu)
            rep.append(SepaConv(in_fi, out_fi))
            rep.append(nn.BatchNorm3d(out_fi))
            filters = out_fi

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SepaConv(filters, filters))
            rep.append(nn.BatchNorm3d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SepaConv(in_fi, out_fi))
            rep.append(nn.BatchNorm3d(out_fi))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)
        
        if strides != 1:
            rep.append(nn.MaxPool3d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip

        return x

class DeConv(nn.Module):
    def __init__(self, in_fi, out_fi):
        super(DeConv, self).__init__()
        self.conv = nn.Conv3d(in_fi, out_fi, 3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(out_fi)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(p=0.25)
        self.upsamp = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x) 
        x = self.dropout(x)
        x = self.upsamp(x)
        return x


class Xception(nn.Module):
    def __init__(self, ich=3):
        super(Xception, self).__init__()

        self.conv1 = nn.Conv3d(ich, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(64)

        self.block1 = Block(64, 128, reps=2, strides=2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, reps=2, strides=2, start_with_relu=False, grow_first=True)
        self.block3 = Block(256, 728, reps=2, strides=2, start_with_relu=False, grow_first=True)

        self.block4 = Block(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True)

        self.block8 = Block(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True)
        
        self.block12 = Block(728, 1024, reps=2, strides=1, start_with_relu=True, grow_first=False)

        self.conv3 = SepaConv(1024, 1536)
        self.bn3 = nn.BatchNorm3d(1536)

        self.conv4 = SepaConv(1536, 2048)
        self.bn4 = nn.BatchNorm3d(2048)

        #Decoder
        self.uconv1 = DeConv(2048, 1024)
        self.uconv2 = DeConv(1024, 256)
        self.uconv3 = DeConv(256, 128)
        self.uconv4 = DeConv(128, 32)

        self.conv_last = nn.Conv3d(32, 1, 3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
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
        
        #Decoder
        x = self.uconv1(x)
        x = self.uconv2(x)
        x = self.uconv3(x)
        x = self.uconv4(x)

        x = self.conv_last(x)

        return self.sigmoid(x)




