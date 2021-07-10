import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AUNet(nn.Module):
    def __init__(self, cfg):
        super(AUNet, self).__init__()
        
        self.template_channel = cfg["UPDATE"].get("TEMPLATE_CHANNEL", 512)
        self.batch = cfg["TRAIN"]["BATCH_SIZE"]

        mid_channel = int(self.template_channel // 8 * 3)
        # 96 in the original UpdateNet
        kernel = cfg["UPDATE"].get("KERNEL_SZ", 1)

        self.shortcut = cfg["UPDATE"].get("SHORTCUT", True)

        self.conv51 = nn.Sequential(nn.Conv2d(self.template_channel, self.template_channel, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(self.template_channel),
                                   nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(self.template_channel, self.template_channel, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(self.template_channel),
                                   nn.ReLU())
        self.conv8 = nn.Sequential(nn.Conv2d(self.template_channel, self.template_channel, 1))

        # adaptive
        # self.similar = nn.Conv2d(self.template_channel, self.template_channel, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(mid_channel*36, mid_channel*36),
            nn.ReLU(True),
            nn.Linear(mid_channel*36, 1),
            nn.Sigmoid()
        )
        # self.fc = nn.Linear(self.batch*512*6*6, 2)

        self._construct_updmod(
            self.template_channel,
            mid_channel, 
            kernel
        )

    def _construct_updmod(
        self, 
        template_channel,
        mid_channel,
        kernel
    ):
        self.a = nn.Conv2d(template_channel*3, mid_channel, kernel_size=kernel)
        self.a_relu = nn.ReLU(inplace=True)
        self.b = nn.Conv2d(mid_channel, template_channel, kernel_size=kernel)
        self.ac = nn.Conv2d(template_channel*3, mid_channel, kernel_size=kernel)
        self.ac_bn = nn.BatchNorm2d(mid_channel)
        self.ac_relu = nn.ReLU(inplace=True)
        
    
    def forward(self, x):
        # x0 = x[:, :self.template_channel, :, :]
        assert type(x) is list
        xcls = self.adaptive(x)
        x, x0, xa = x[0], x[1], x[2]
        
        # xi = torch.zeros_like(x)      
        # for i in range(x.shape[0]):
        #     xi[i] = xcls[i]*x[i] + (1-xcls[i])*xa[i]
        # x = xi

        xi = torch.zeros_like(x)
        for i in range(x.shape[0]):
            # xi[i] = xcls[i,0]*xp[i] + xcls[i,1]*x[i]
            if xcls[i] > 0.6:
                xi[i] = x[i]
            else:
                return [xa, xcls]
        x = xi
        # x = xcls * x + (1-xcls) * xp
        x = torch.cat((x0, xa, x),1)
        x = self.a(x)
        x = self.a_relu(x)
        x = self.b(x)
        if x0 is not None and self.shortcut:
            x = x + x0
        output = [x, xcls]
        return output

    def adaptive(self, x):
        # previous frame to predict class
        assert type(x) is list
        x, x0, xa = x[0], x[1], x[2]
        xcls = torch.cat((x0, xa, x),1)
        xcls = self.ac(xcls)
        xcls = self.ac_bn(xcls)
        xcls = self.ac_relu(xcls)
        # xcls = self.avgpool(xcls)
        xcls = xcls.view(xcls.size(0), -1)
        xcls = self.classifier(xcls)
        
        return xcls