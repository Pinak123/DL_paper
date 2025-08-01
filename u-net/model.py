import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))
            self.bottleneck = DoubleConv(features[-1], features[-1]*2)
            self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # x shape: (N, in_channels, H, W)
        skip_connections = []
        for down in self.downs:
            x = down(x)
            # After DoubleConv: (N, feature, H, W)
            skip_connections.append(x)
            x = self.pool(x)
            # After MaxPool2d: (N, feature, H/2, W/2)

        x = self.bottleneck(x)
        # After bottleneck: (N, features[-1]*2, H/16, W/16) if 4 downsamples
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            # After ConvTranspose2d (upsample): (N, feature, H*2, W*2)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
                # After resize: (N, feature, skip_H, skip_W)
            concat_skip = torch.cat((skip_connection, x), dim=1)
            # After concat: (N, feature*2, H, W)
            x = self.ups[idx+1](concat_skip)
            # After DoubleConv: (N, feature, H, W)
        # Final output: (N, out_channels, H, W)
        return self.final_conv(x)

def test():
    x = torch.randn((3, 1, 160, 160))
    model = UNet(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)


if __name__ == "__main__":
    test()