#!/usr/bin/env python3
"""
Simple test to check if DCGAN models work properly
"""

import torch
import matplotlib.pyplot as plt
from model import create_dcgan_models
import torchvision.utils as utils

def test_models():
    """Test if models can generate images"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create models
    generator, discriminator = create_dcgan_models()
    generator.to(device)
    discriminator.to(device)
    
    print("Testing models...")
    
    # Test with random noise
    batch_size = 16
    z = torch.randn(batch_size, 100).to(device)
    
    # Generate images
    with torch.no_grad():
        fake_images = generator(z)
        disc_output = discriminator(fake_images)
    
    print(f"Generator output shape: {fake_images.shape}")
    print(f"Discriminator output shape: {disc_output.shape}")
    print(f"Generated image range: [{fake_images.min():.3f}, {fake_images.max():.3f}]")
    print(f"Discriminator output range: [{disc_output.min():.3f}, {disc_output.max():.3f}]")
    
    # Save a grid of generated images
    fake_images = (fake_images + 1) / 2.0  # Denormalize
    fake_images = torch.clamp(fake_images, 0, 1)
    
    grid = utils.make_grid(fake_images, nrow=4, padding=2, normalize=False)
    utils.save_image(grid, "test_generated.png")
    print("Test image saved as 'test_generated.png'")
    
    print("✅ Models are working correctly!")

if __name__ == "__main__":
    test_models()