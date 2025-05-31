import torch

# Define your UNet architecture with the correct feature dimensions
class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv1x1 = torch.nn.Conv2d(in_channels, in_channels//2, kernel_size=1)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = torch.nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if hasattr(self, 'conv1x1'):
            x1 = self.conv1x1(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = torch.nn.functional.pad(x1, [diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(torch.nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True, features=32):  # Changed features to 32
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, features)
        self.down1 = Down(features, features*2)
        self.down2 = Down(features*2, features*4)
        self.down3 = Down(features*4, features*8)
        self.down4 = Down(features*8, features*16)
        self.up1 = Up(features*16, features*8, bilinear)
        self.up2 = Up(features*8, features*4, bilinear)
        self.up3 = Up(features*4, features*2, bilinear)
        self.up4 = Up(features*2, features, bilinear)
        self.outc = torch.nn.Conv2d(features, n_classes, kernel_size=1)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        if logits.size()[2:] != input_size:
            logits = torch.nn.functional.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        return torch.sigmoid(logits)