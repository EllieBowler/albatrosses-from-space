from torch import nn
import torch


class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(inconv, self).__init__()
        self.conv = double_conv(in_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(down, self).__init__()

        if dropout:
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_channels, out_channels),
                nn.Dropout(0.5)
            )
        else:
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_channels, out_channels)
            )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 2, stride=2)

        self.conv = double_conv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = (x2.size()[2] - x1.size()[2]) // 2
        diffX = (x2.size()[3] - x1.size()[3]) // 2

        crop_x2 = x2[:, :, diffY:(diffY + x1.size()[2]), diffX:(diffX + x1.size()[3])]

        x = torch.cat([crop_x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.conv(x)
        return x