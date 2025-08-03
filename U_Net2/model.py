import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.inc = self.double_conv(in_channels, 64)
        self.down1 = self.double_conv(64, 128)
        self.down2 = self.double_conv(128, 256)
        self.down3 = self.double_conv(256, 512)
        self.down4 = self.double_conv(512, 1024)
        
        # Proper upsampling blocks with skip connections
        self.up1 = self.up_block(1024, 512)
        self.up2 = self.up_block(512, 256)
        self.up3 = self.up_block(256, 128)
        self.up4 = self.up_block(128, 64)
        
        # Decoder convolution layers (for processing concatenated features)
        self.up_conv1 = self.double_conv(1024, 512)  # 1024 (512+512) -> 512
        self.up_conv2 = self.double_conv(512, 256)   # 512 (256+256) -> 256
        self.up_conv3 = self.double_conv(256, 128)   # 256 (128+128) -> 128
        self.up_conv4 = self.double_conv(128, 64)    # 128 (64+64) -> 64
        
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def up_block(self, in_channels, out_channels):
        """Upsampling block with skip connection handling"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder path (contracting) - save skip connections BEFORE maxpooling
        x1 = self.inc(x)         # 64 channels, 256x256
        
        x2 = self.maxpool(x1)    # 64 channels, 128x128
        x2 = self.down1(x2)      # 128 channels, 128x128
        
        x3 = self.maxpool(x2)    # 128 channels, 64x64
        x3 = self.down2(x3)      # 256 channels, 64x64
        
        x4 = self.maxpool(x3)    # 256 channels, 32x32
        x4 = self.down3(x4)      # 512 channels, 32x32
        
        x5 = self.maxpool(x4)    # 512 channels, 16x16
        x5 = self.down4(x5)      # 1024 channels, 16x16
        
        # Decoder path (expanding) with skip connections
        x = self.up1(x5)         # 1024 -> 512, 32x32
        x = self._concat_skip(x, x4)  # Skip connection with x4 (512 channels, 32x32)
        x = self.up_conv1(x)     # 1024 (512+512) -> 512
        
        x = self.up2(x)          # 512 -> 256, 64x64
        x = self._concat_skip(x, x3)  # Skip connection with x3 (256 channels, 64x64)
        x = self.up_conv2(x)     # 512 (256+256) -> 256
        
        x = self.up3(x)          # 256 -> 128, 128x128
        x = self._concat_skip(x, x2)  # Skip connection with x2 (128 channels, 128x128)
        x = self.up_conv3(x)     # 256 (128+128) -> 128
        
        x = self.up4(x)          # 128 -> 64, 256x256
        x = self._concat_skip(x, x1)  # Skip connection with x1 (64 channels, 256x256)
        x = self.up_conv4(x)     # 128 (64+64) -> 64
        
        x = self.outc(x)
        return torch.sigmoid(x)  # Add sigmoid for proper probability output
    
    def _concat_skip(self, x, skip):
        """Handle size mismatch and concatenate skip connection"""
        # Handle input size differences
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        
        if diffY > 0 or diffX > 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel dimension
        return torch.cat([skip, x], dim=1)
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = UNet(1, 1).to(device)
    x = torch.randn(1, 1, 256, 256).to(device)
    
    print(f"Using device: {device}")
    print(f"Input shape: {x.shape}")
    
    output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"✅ Model works correctly!")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    print("\n📊 Model Architecture Summary:")
    print("=" * 50)
    summary(model, (1, 256, 256), device=device_str)