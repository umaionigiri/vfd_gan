
import torch
import torch.nn as nn


class C2plus1d_Block(nn.Module):
    def __init__(self, in_ch, out_ch, k=3):
        super(C2plus1d_Block, self).__init__()
        
        self.conv = nn.Conv3d(in_ch, out_ch, 1, stride=1)

        self.spaceconv = nn.Conv3d(in_ch, in_ch, (1, k, k), stride=1, padding=(0, 1, 1), dilation=1, bias=False)
        self.pointwise = nn.Conv3d(in_ch, out_ch, (3, 1, 1), stride=1, padding=(1, 0, 0), dilation=1, bias=False)

        self.bn1 = nn.BatchNorm3d(in_ch)
        self.bn2 = nn.BatchNorm3d(out_ch)

        self.avgpool = nn.AvgPool3d(2)
        self.dropout = nn.Dropout(p=0.25)
        self.upsamp = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.relu = nn.ReLU(inplace=True)
        self.conv_last = nn.Conv3d(out_ch+out_ch, out_ch, 3, stride=1, padding=1, dilation=1, bias=False)

    def forward(self, x, down_samp=False):
        
        inp = x
        
        x = self.spaceconv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)

        if down_samp: 
            x = self.avgpool(x)
            inp = self.conv(inp)
            inp = self.avgpool(inp)
        else: 
            x = self.upsamp(x)
            inp = self.dropout(inp)
            inp = self.upsamp(inp)
            inp = self.conv(inp)

        x = torch.cat([x, inp], dim=1)
        x = self.conv_last(x)

        return x

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        self.down_sep1 = C2plus1d_Block(3, 64)
        self.down_sep2 = C2plus1d_Block(64, 128)
        self.down_sep3 = C2plus1d_Block(128, 256)
        self.down_sep4 = C2plus1d_Block(256, 512)
        
        self.up_sep1 = C2plus1d_Block(512, 256)
        self.up_sep2 = C2plus1d_Block(256+256, 256)
        self.up_sep3 = C2plus1d_Block(256+128, 128)
        self.up_sep4 = C2plus1d_Block(128+64, 64)

        self.conv_last = nn.Conv3d(64, 1, 3, stride=1, padding=1, bias=False)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        
        # Encoder
        down_sep1 = self.down_sep1(x, down_samp=True) # 64x8x64x64
        down_sep2 = self.down_sep2(down_sep1, down_samp=True) # 128x4x32x32
        down_sep3 = self.down_sep3(down_sep2, down_samp=True) # 256x2x16x16
        down_sep4 = self.down_sep4(down_sep3, down_samp=True) # 512x1x8x8
        
        #Decoder
        up_sep1 = self.up_sep1(down_sep4, down_samp=False) # 256x2x16x16
        x = torch.cat([up_sep1, down_sep3], dim=1) # 512x2x16x16
        up_sep2 = self.up_sep2(x, down_samp=False) # 256x4x32x32
        x = torch.cat([up_sep2, down_sep2], dim=1) # 256+128x4x32x32
        up_sep3 = self.up_sep3(x, down_samp=False) # 128x8x64x64
        x = torch.cat([up_sep3, down_sep1], dim=1) # 128+64x8x64x64
        up_sep4 = self.up_sep4(x, down_samp=False) # 64x16x128x128

        x = self.conv_last(up_sep4) # 1x16x128x128

        return self.sigmoid(x)

