'''

'''

import torch.nn as nn

import torch
from torch.nn import functional as F

ds = torch.distributions
def cuda(tensor, is_cuda):
    if is_cuda:
        return tensor.cuda()
    else:
        return tensor


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MSCNN(nn.Module):
    def __init__(self, channel_msi):
        super(MSCNN, self).__init__()
        self.conv3x3_1 = nn.Sequential(
                                       nn.Conv2d(channel_msi, 128, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU(inplace=False)
                                      )

        self.conv3x3_2 = nn.Sequential(
                                       nn.Conv2d(128, 64, 1),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(inplace=False)
                                       )

        self.conv3x3_3 = nn.Sequential(
                                        nn.Conv2d(64, 64, 3, 1, 1, bias=False),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(inplace=False),
                                        nn.MaxPool2d(2, 2),
                                        nn.ReLU(inplace=False)
                                      )

        self.SELayer = SELayer(64)


    def forward(self, x):
        x = self.conv3x3_1(x)
        x = self.conv3x3_2(x)
        x = self.SELayer(x)
        x = self.conv3x3_3(x)
        return x

class PCNN(nn.Module):
    def __init__(self, channel_pan):
        super(PCNN, self).__init__()

        self.conv3x3_1 = nn.Sequential(
                                       nn.Conv2d(channel_pan, 16, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(16),
                                       nn.ReLU(inplace=False)
                                    )

        self.conv3x3_2 = nn.Sequential(
                                       nn.Conv2d(16, 32, 1),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(inplace=False)
                                     )

        self.conv3x3_3 = nn.Sequential(
                                            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
                                            nn.BatchNorm2d(64),
                                            nn.ReLU(inplace=False),
                                            nn.MaxPool2d(2, 2),
                                            nn.ReLU(inplace=False)
                                      )
        self.SELayer = SELayer(32)

    def forward(self, x):
        x = self.conv3x3_1(x)
        x = self.conv3x3_2(x)
        x = self.SELayer(x)
        x = self.conv3x3_3(x)
        return x


class NormInteraction(nn.Module):
    def __init__(self, inplanes=128):
        super(NormInteraction, self).__init__()

        self.conv = nn.Conv2d(inplanes * 2, inplanes, kernel_size=1, stride=1)
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=False))


    def forward(self, x1, x2):
        concat = torch.cat([x1, x2], 1)
        gate = torch.sigmoid(self.conv(concat))
        p1 = x1 * gate
        p2 = x2 * (1 - gate)
        m_1 = torch.mean(gate)
        p = torch.matmul(p1, p2)
        p = self.conv3(p)
        return p, m_1


class IBfusion(nn.Module):
    def __init__(self, dim, num_classes, inner=128):
        super(IBfusion, self).__init__()
        self.d_l = dim

        self.encoder = nn.Sequential(nn.Linear(self.d_l, inner),
                                 nn.ReLU(inplace=True))

        self.fc_mu = nn.Linear(inner, self.d_l)
        self.fc_std = nn.Linear(inner, self.d_l)

        self.decoder = nn.Linear(self.d_l, num_classes)


    def encode(self, x):
        x = self.encoder(x)
        return self.fc_mu(x), F.softplus(self.fc_std(x) - 5, beta=1)

    def decode(self, z):
        return self.decoder(z)

    def reparameterise(self, mu, std):
        eps = torch.randn_like(std)
        return mu + std*eps

    def forward(self, x):
        mu, std = self.encode(x)
        z = self.reparameterise(mu, std)
        output = self.decode(z)

        return output, mu, std

class MIBfusion(nn.Module):
    def __init__(self, in_channels1, in_channels2, num_classes):
        super(MIBfusion, self).__init__()

        self.net1 = MSCNN(in_channels1)
        self.net2 = PCNN(in_channels2)

        self.interaction = NormInteraction(64)

        self.m = torch.tensor(0.5)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))


        self.f1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
        )

        self.f2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
        )

        self.ff = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(128, num_classes),
        )

        self.IB_fusion = IBfusion(128, 128, num_classes)
        self.ii = torch.tensor(0.0)


    def forward(self, HS, SAR):

        self.ii = self.ii + 1

        branch1 = self.net1(HS)
        branch2 = self.net2(SAR)

        if self.training and self.ii % 1 == 0:
            o = torch.bernoulli(1-self.m)
            size = branch1[0].size()
            indices = torch.randint(0, branch1.shape[0], [int(branch1.shape[0] * 0.7), 1])
            if o == 0:
                for ind in indices:
                    rand_g = torch.randn(size, dtype=torch.float32).to(branch1.device)
                    branch1[ind] += rand_g
            else:
                for ind in indices:
                    rand_g = torch.randn(size, dtype=torch.float32).to(branch2.device)
                    branch2[ind] += rand_g


        align_fusion, self.m = self.interaction(branch1, branch2)
        branch1_1 = self.f1(branch1)
        branch2_1 = self.f2(branch2)
        align_fusion = torch.cat([branch1_1, align_fusion, branch2_1], 1)
        align_fusion = self.ff(align_fusion)


        x = self.avg_pool(align_fusion)
        x = torch.flatten(x, 1)
        out, mu, std = self.IB_fusion(x)
        return out, mu, std


class IBloss_R(nn.Module):
    def __init__(self):
        super(IBloss_R, self).__init__()
        self.alpha = 1e-3

    def forward(self, output, y):
        logit, mu, std = output
        class_loss = F.cross_entropy(logit, y.squeeze())
        info_loss = 0.5 * torch.mean(mu.pow(2) + std.pow(2) - 2 * std.log() - 1)
        total_loss = class_loss + self.alpha * info_loss
        return total_loss

if __name__ == "__main__":
    HS = torch.randn(12, 244, 7, 7)
    SAR = torch.randn(12, 4, 7, 7)
    y = torch.randint(0, 7, (1, 12))
    grf_net = MIBfusion(in_channels1=244, in_channels2=4, num_classes=7)
    out = grf_net(HS, SAR)
    LossFunc = IBloss_R()
    loss = LossFunc(out, y)
