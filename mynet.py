import torch
import torch.nn as nn
import torch.nn.functional as F

class mynet_d(nn.Module):
    def __init__(self):
        super(mynet_d, self).__init__()
        netArc_checkpoint = '/home/amax/LZG/simswap/arcface_model/arcface_checkpoint.tar'
        netArc_checkpoint = torch.load(netArc_checkpoint, map_location=torch.device("cpu"))
        self.netArc = netArc_checkpoint['model'].module
        self.netArc = self.netArc
        self.netArc.eval()
        self.netArc.requires_grad_(False)

        self.yc = nn.Sequential(nn.Linear(512,256),nn.BatchNorm1d(256),nn.LeakyReLU(),
                                 nn.Linear(256,128),nn.BatchNorm1d(128),nn.LeakyReLU(),
                                 nn.Linear(128,64))

    def forward(self,x,y): ## x : [b,3,224,224]
        x = F.interpolate(x, size=(112, 112), mode='bicubic')
        y = F.interpolate(y, size=(112, 112), mode='bicubic')
        # print(x.dtype)
        x_id = self.netArc(x)
        y_id = self.netArc(y)
        # x_id = torch.reshape(x_id,(4,512,1))
        out = self.yc(x_id - y_id)
        out = torch.reshape(out,(4,64))
        return out

if __name__ == '__main__':

    mynet = mynet_d().cuda()
    x = torch.rand(4,3,224,224).cuda()
    out = mynet(x)
    print(out)
    print(out.shape)


