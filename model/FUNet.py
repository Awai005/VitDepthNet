import sys
sys.path.append('')
import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import functional as F

from VitTransformer import *

class BoxFilter(nn.Module):
    def __init__(self, kernel_size=3):
        super(BoxFilter, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2  # assuming odd kernel size

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=self.kernel_size, stride=1, padding=self.padding)


class DAIFNet(nn.Module):
    def __init__(self, input_ch, output_ch, W=16, D=4, ret_bottleneck=False):
        super(DAIFNet, self).__init__()
        self.conv_down = nn.ModuleList([self.convblock(input_ch, W)] + [self.convblock(W * (2**i), W * (2**i)) for i in range(1, D)])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(W * (2**D), W * (2**D), kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv_up = nn.ModuleList([self.upconvblock(W * (2**i), W * (2**i) // 2) for i in range(D, 0, -1)])
        self.conv_joint = nn.ModuleList([self.convblock(W * (2**i), W * (2**i // 2)) for i in range(D, 0, -1)])


        self.box_filter = BoxFilter().cuda()
        self.transformer = VisionTransformerDepthEstimator_bn(patch_size=4, projection_dim=256, num_heads=8, transformer_layers=8)


        self.conv_out = nn.Sequential(
            nn.Conv2d(W, W, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(W, output_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.ret_bottleneck = ret_bottleneck
        
    def convblock(self, in_ch,out_ch):
        block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),            
            nn.ReLU(),
        )
        return block
    
    def upconvblock(self,in_ch,out_ch):
        block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        return block

    def forward(self, x):
        B, FS, C, H, W = x.shape
        h = x.view(B*FS, C, H, W) # BxFS C H W
        h_s = []
        
        for i, l in enumerate(self.conv_down):
            #print("downsampling", h.shape)
            h = self.conv_down[i](h) # BxFS C H/2**i W/2**i
            pool_h = F.max_pool2d(h, kernel_size=2, stride=2, padding=0) # BxFS C H/2**(i+1) W/2**(i+1)
            w_h = h.view(B, FS, *h.shape[-3:]) # B FS C H/2**i W/2**i
            h_s.append(torch.max(w_h, dim=1)[0]) # B C H/2**i W/2**i
            # Global Operation
            stack_pool = pool_h.view(B, FS, *pool_h.shape[-3:])
            pool_max = torch.max(stack_pool, dim=1)[0].unsqueeze(1).expand_as(stack_pool).contiguous().view(B*FS, *pool_h.shape[-3:])
            h = torch.cat([pool_h, pool_max], dim=1)
        
        #print("Bottleneck", h.shape)
        h = self.bottleneck(h)
        
        h = self.transformer(h)

        w_h = h.view(B, FS, *h.shape[-3:]) # B FS C H W
        h = torch.max(w_h, dim=1)[0]
        

        for i, l in enumerate(self.conv_up):
            #print("upsampling", h.shape)
            h = self.conv_up[i](h)
            skip_h = h_s.pop(-1)
            h = self.conv_joint[i](torch.cat([h, skip_h], dim=1))
            
            h = self.box_filter(h)   
        
        output = self.conv_out(h)
        if self.ret_bottleneck:
            return output, w_h
        else:
            return output
