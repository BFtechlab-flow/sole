import torch.nn as nn
import torch.nn.functional as F
import timm

class ConvBlock(nn.Module):
    def __init__(self,cin,cout):
        super().__init__()
        g=8 if cout%8==0 else 1
        self.net=nn.Sequential(nn.Conv2d(cin,cout,3,padding=1,bias=False),nn.GroupNorm(g,cout),nn.SiLU(),
                               nn.Conv2d(cout,cout,3,padding=1,bias=False),nn.GroupNorm(g,cout),nn.SiLU())
    def forward(self,x): return self.net(x)

class FILANet(nn.Module):
    def __init__(self,encoder="convnext_tiny.fb_in1k",pretrained=True,fpn_channels=128,in_chans=4):
        super().__init__()
        self.encoder=timm.create_model(encoder,pretrained=pretrained,in_chans=in_chans,
                                       features_only=True,out_indices=(0,1,2,3))
        cs=self.encoder.feature_info.channels()
        self.lat=nn.ModuleList([nn.Conv2d(c,fpn_channels,1) for c in cs])
        self.ref=nn.ModuleList([ConvBlock(fpn_channels,fpn_channels) for _ in cs])
        self.fuse=ConvBlock(fpn_channels*len(cs),fpn_channels)
        self.heads=nn.ModuleDict({k:nn.Conv2d(fpn_channels,1,1)
            for k in ("region","centerline","boundary","distance")})
    def forward(self,x):
        size=x.shape[-2:]; f=self.encoder(x); p=[None]*len(f); p[-1]=self.lat[-1](f[-1])
        for i in range(len(f)-2,-1,-1):
            p[i]=self.lat[i](f[i])+F.interpolate(p[i+1],size=f[i].shape[-2:],mode="bilinear",align_corners=False)
        p=[self.ref[i](p[i]) for i in range(len(p))]
        base=p[0].shape[-2:]
        z=self.fuse(__import__("torch").cat([F.interpolate(q,size=base,mode="bilinear",align_corners=False) for q in p],1))
        z=F.interpolate(z,size=size,mode="bilinear",align_corners=False)
        return {k:h(z) for k,h in self.heads.items()}
