""" Full assembly of the parts to form the complete network """
import torch.nn as nn
from unet_parts import *
import torch

class E2D(nn.Module):
    def __init__(self, n_channels, n_dim, bilinear=False):
        super(E2D, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, n_dim)
        self.down1 = Down(n_dim,n_dim*2)
        self.down2 = Down(n_dim*2, n_dim*4)
        self.down3 = Down(n_dim*4, n_dim*8)

        self.up3   = Up2(n_dim*8,n_dim*4,bilinear,dp=True)
        self.up2   = Up2(n_dim*4, n_dim*2,bilinear,dp=True)
        self.up1   = Up2(n_dim*2, n_dim,bilinear,dp=True)
        # factor = 2 if bilinear else 1
        # self.up3 = Up(256, 128 // factor, bilinear,dp=True)
        # self.up4 = Up(128, 64, bilinear,dp=True)
        self.outc = OutConv(n_dim, n_channels)
        self.dropout = nn.Dropout2d(0.5)

    def forward(self, x):
        # (batch,in_channels,height,width) -> (batch,n_dim,height,width) 
        x = self.inc(x)
        x = self.dropout(x)

        # (batch,n_dim,height,width) -> (batch,n_dim*2,height//2,width//2) 
        x = self.down1(x)
        x = self.dropout(x)

        # (batch,n_dim*2,height//2,width//2) -> (batch,n_dim*4,height//4,width//4) 
        x = self.down2(x)
        x = self.dropout(x)

        # (batch,n_dim*4,height//4,width//4) -> (batch,n_dim*8,height//8,width//8) 
        x = self.down3(x)
        # x = self.dropout(x)


        # (batch,n_dim*8,height//8,width//8) -> (batch,n_dim*4,height//4,width//4) 
        x = self.up3(x)

        # (batch,n_dim*4,height//4,width//4) -> (batch,n_dim*2,height//2,width//2) 
        x = self.up2(x)

        # (batch,n_dim*2,height//2,width//2) -> (batch,n_dim,height,width) 
        x = self.up1(x)
        
        # (batch, n_dim, height,width) -> (batch,n_channels, height, width)
        x =self.outc(x)
        return x


if __name__=="__main__":
    model = E2D(n_channels=1,n_dim=32)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters: {:,}".format(num_params))
    x = torch.rand(1, 1, 72, 72)
    print(model(x).shape)
    print(model)
    print("Done!")