import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=False,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PatchDiscriminator(nn.Module):

    def __init__(self, in_channels: int = 3, features: list[int] = [64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels * 2, features[0], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers: list[nn.Module] = []
        in_c = features[0]
        for feature in features[1:]:
            # stride=2 except for the last block which often uses stride=1
            stride = 1 if feature == features[-1] else 2
            layers.append(CNNBlock(in_c, feature, stride=stride))
            in_c = feature

        layers.append(
            nn.Conv2d(in_c, 1, kernel_size=4, stride=1, padding=1)
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = torch.cat([x, y], dim=1)
        z = self.initial(z)
        z = self.model(z)
        return z


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, down: bool = True, use_act: bool = True, use_dropout: bool = False):
        super().__init__()
        if down:
            self.block = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    padding_mode="reflect",
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True) if use_act else nn.Identity(),
            )
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True) if use_act else nn.Identity(),
            )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5) if use_dropout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        x = self.dropout(x)
        return x


class UNetGenerator(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 3, features: list[int] = [64, 128, 256, 512, 512, 512, 512]):
        super().__init__()
        # Encoder (down) blocks
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, features[0], 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
        )  # no BN in first layer per paper
        self.down2 = ConvBlock(features[0], features[1], down=True)
        self.down3 = ConvBlock(features[1], features[2], down=True)
        self.down4 = ConvBlock(features[2], features[3], down=True)
        self.down5 = ConvBlock(features[3], features[4], down=True)
        self.down6 = ConvBlock(features[4], features[5], down=True)
        self.down7 = ConvBlock(features[5], features[6], down=True)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[6], features[6], 4, 2, 1, padding_mode="reflect"),
            nn.ReLU(inplace=True),
        )

        # Decoder (up) blocks
        self.up1 = ConvBlock(features[6], features[6], down=False, use_dropout=True)
        self.up2 = ConvBlock(features[6] * 2, features[5], down=False, use_dropout=True)
        self.up3 = ConvBlock(features[5] * 2, features[4], down=False, use_dropout=True)
        self.up4 = ConvBlock(features[4] * 2, features[3], down=False)
        self.up5 = ConvBlock(features[3] * 2, features[2], down=False)
        self.up6 = ConvBlock(features[2] * 2, features[1], down=False)
        self.up7 = ConvBlock(features[1] * 2, features[0], down=False)
        self.final_up = nn.ConvTranspose2d(features[0] * 2, out_channels, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        bottleneck = self.bottleneck(d7)

        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, d7], dim=1))
        u3 = self.up3(torch.cat([u2, d6], dim=1))
        u4 = self.up4(torch.cat([u3, d5], dim=1))
        u5 = self.up5(torch.cat([u4, d4], dim=1))
        u6 = self.up6(torch.cat([u5, d3], dim=1))
        u7 = self.up7(torch.cat([u6, d2], dim=1))
        out = self.final_up(torch.cat([u7, d1], dim=1))
        return self.tanh(out)


__all__ = [
    "PatchDiscriminator",
    "UNetGenerator",
]


