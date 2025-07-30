import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels_img=1, features_d=64):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input: N x channels_img x 28 x 28
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # State: features_d x 14 x 14
            self._block(features_d, features_d * 2, 4, 2, 1),
            # State: features_d*2 x 7 x 7
            self._block(features_d * 2, features_d * 4, 3, 2, 1),
            # State: features_d*4 x 4 x 4
            nn.Conv2d(features_d * 4, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid(),
            # Output: 1 x 1 x 1
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),  # Re-enable BatchNorm for stability
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, channels_noise=100, channels_img=1, features_g=64):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 8, 4, 1, 0),  # img: 4x4
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 8x8
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 16x16
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 32 x 32
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),  # Re-enable BatchNorm for stability
            nn.ReLU(),
        )

    def forward(self, x):
        # Reshape z to (batch_size, z_dim, 1, 1)
        x = x.view(x.size(0), x.size(1), 1, 1)
        output = self.net(x)
        # Crop to 28x28 for MNIST (from 32x32)
        return output[:, :, 2:30, 2:30]


def weights_init(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


def create_dcgan_models(z_dim=100, img_size=28, num_channels=1, num_classes=10):
    """
    Create and initialize DCGAN Generator and Discriminator for MNIST
    """
    # Create models
    generator = Generator(channels_noise=z_dim, channels_img=num_channels, features_g=64)
    discriminator = Discriminator(channels_img=num_channels, features_d=64)
    
    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    return generator, discriminator
