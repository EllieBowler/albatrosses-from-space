from .unet_parts import *
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear, dropout):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 32)
        self.down1 = down(32, 64, dropout)
        self.down2 = down(64, 128, dropout)
        self.down3 = down(128, 256, dropout)
        self.down4 = down(256, 256, dropout)
        self.up1 = up(512, 128, bilinear)
        self.up2 = up(256, 64, bilinear)
        self.up3 = up(128, 32, bilinear)
        self.up4 = up(64, 32, bilinear)
        self.outc = outconv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

    def pretrained_parameters(self):
        return []

    def new_parameters(self):
        return self.parameters()


class vgg16encoder(nn.Module):
    def __init__(self, vgg16, n_channels):
        super(vgg16encoder, self).__init__()
        self.conv1_1 = vgg16.features[0]
        self.conv1_2 = vgg16.features[2]
        self.pool1 = vgg16.features[4]

        self.conv2_1 = vgg16.features[5]
        self.conv2_2 = vgg16.features[7]
        self.pool2 = vgg16.features[9]

        self.conv3_1 = vgg16.features[10]
        self.conv3_2 = vgg16.features[12]
        self.conv3_3 = vgg16.features[14]
        self.pool3 = vgg16.features[16]

        self.conv4_1 = vgg16.features[17]
        self.conv4_2 = vgg16.features[19]
        self.conv4_3 = vgg16.features[21]
        self.pool4 = vgg16.features[23]

        self.conv5_1 = vgg16.features[24]
        self.conv5_2 = vgg16.features[26]
        self.conv5_3 = vgg16.features[28]

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x1 = x = F.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = F.relu(self.conv2_1(x))
        x2 = x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x3 = x = F.relu(self.conv3_3(x))
        x = self.pool3(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x4 = x = F.relu(self.conv4_3(x))
        x = self.pool4(x)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x5 = F.relu(self.conv5_3(x))

        return x1, x2, x3, x4, x5

    def expand_input(self, n_channels):
        # The shape of w is (out_channels, 3, 3, 3) (with last two kernel width and height)
        # Need to extend dimension 1 by the required number of extra channels
        w = self.conv1_1.weight
        w_x = torch.randn(w.shape[0], n_channels - w.shape[1], w.shape[2], w.shape[3]).cuda() * torch.std(w) + torch.mean(w)
        w_new = torch.cat([w, w_x], dim=1)
        self.conv1_1.weight = nn.Parameter(w_new)


class vgg16_UNet(nn.Module):
    def __init__(self, vgg16, n_channels, n_classes, bilinear, dropout):
        super(vgg16_UNet, self).__init__()

        if n_channels == 3:
            print('Inputting RGB to vgg16')
        else:
            print('Taking {} channel input'.format(n_channels))

        self.encoder = vgg16encoder(vgg16, n_channels)

        self.up1 = up(1024, 512, bilinear)
        self.up2 = up(768, 256, bilinear)
        self.up3 = up(384, 128, bilinear)
        self.up4 = up(192, 32, bilinear)
        self.outc = outconv(32, n_classes)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

    def pretrained_parameters(self):
        return self.encoder.parameters()

    def new_parameters(self):
        pretrained_ids = [id(p) for p in self.pretrained_parameters()]

        return [p for p in self.parameters() if id(p) not in pretrained_ids]
