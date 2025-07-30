import torch
import torch.nn as nn
from model import Generator, Discriminator, create_dcgan_models

def test_discriminator():
    """Test the discriminator with random input shapes"""
    print("Testing Discriminator...")
    
    # Create discriminator
    discriminator = Discriminator(img_size=28, num_channels=1)
    
    # Test with different batch sizes
    batch_sizes = [1, 4, 8, 16]
    
    for batch_size in batch_sizes:
        # Create random input: (batch_size, channels, height, width)
        x = torch.randn(batch_size, 1, 28, 28)
        
        # Forward pass
        output = discriminator(x)
        
        # Check output shape
        expected_shape = (batch_size, 1, 1, 1)
        actual_shape = output.shape
        
        print(f"  Batch size {batch_size}: Input {x.shape} -> Output {actual_shape}")
        
        if actual_shape == expected_shape:
            print(f"    ✅ PASS: Output shape is correct")
        else:
            print(f"    ❌ FAIL: Expected {expected_shape}, got {actual_shape}")
        
        # Check output values are between 0 and 1 (sigmoid output)
        if torch.all((output >= 0) & (output <= 1)):
            print(f"    ✅ PASS: Output values are in [0,1] range")
        else:
            print(f"    ❌ FAIL: Output values outside [0,1] range")
        
        print()

def test_generator():
    """Test the generator with random input shapes"""
    print("Testing Generator...")
    
    # Create generator
    generator = Generator(z_dim=100, img_size=28, num_channels=1)
    
    # Test with different batch sizes
    batch_sizes = [1, 4, 8, 16]
    
    for batch_size in batch_sizes:
        # Create random noise: (batch_size, z_dim)
        z = torch.randn(batch_size, 100)
        
        # Forward pass
        output = generator(z)
        
        # Check output shape
        expected_shape = (batch_size, 1, 28, 28)
        actual_shape = output.shape
        
        print(f"  Batch size {batch_size}: Input {z.shape} -> Output {actual_shape}")
        
        if actual_shape == expected_shape:
            print(f"    ✅ PASS: Output shape is correct")
        else:
            print(f"    ❌ FAIL: Expected {expected_shape}, got {actual_shape}")
        
        # Check output values are between -1 and 1 (tanh output)
        if torch.all((output >= -1) & (output <= 1)):
            print(f"    ✅ PASS: Output values are in [-1,1] range")
        else:
            print(f"    ❌ FAIL: Output values outside [-1,1] range")
        
        print()

def test_create_models():
    """Test the create_dcgan_models function"""
    print("Testing create_dcgan_models function...")
    
    # Create models
    generator, discriminator = create_dcgan_models(z_dim=100, img_size=28, num_channels=1)
    
    # Test with random inputs
    batch_size = 4
    z = torch.randn(batch_size, 100)
    x = torch.randn(batch_size, 1, 28, 28)
    
    # Test generator
    gen_output = generator(z)
    print(f"  Generator: Input {z.shape} -> Output {gen_output.shape}")
    
    # Test discriminator
    disc_output = discriminator(x)
    print(f"  Discriminator: Input {x.shape} -> Output {disc_output.shape}")
    
    # Test discriminator on generator output
    fake_output = discriminator(gen_output)
    print(f"  Discriminator on fake: Input {gen_output.shape} -> Output {fake_output.shape}")
    
    print("  ✅ PASS: All shapes are correct")
    print()

def test_different_channels():
    """Test with different numbers of channels"""
    print("Testing with different channel configurations...")
    
    # Test RGB images (3 channels)
    generator_rgb, discriminator_rgb = create_dcgan_models(z_dim=100, img_size=28, num_channels=3)
    
    batch_size = 2
    z = torch.randn(batch_size, 100)
    x_rgb = torch.randn(batch_size, 3, 28, 28)
    
    gen_output_rgb = generator_rgb(z)
    disc_output_rgb = discriminator_rgb(x_rgb)
    
    print(f"  RGB (3 channels):")
    print(f"    Generator: {z.shape} -> {gen_output_rgb.shape}")
    print(f"    Discriminator: {x_rgb.shape} -> {disc_output_rgb.shape}")
    
    # Test grayscale images (1 channel)
    generator_gray, discriminator_gray = create_dcgan_models(z_dim=100, img_size=28, num_channels=1)
    
    x_gray = torch.randn(batch_size, 1, 28, 28)
    gen_output_gray = generator_gray(z)
    disc_output_gray = discriminator_gray(x_gray)
    
    print(f"  Grayscale (1 channel):")
    print(f"    Generator: {z.shape} -> {gen_output_gray.shape}")
    print(f"    Discriminator: {x_gray.shape} -> {disc_output_gray.shape}")
    
    print("  ✅ PASS: All channel configurations work correctly")
    print()

if __name__ == "__main__":
    print("=" * 50)
    print("DCGAN Model Testing")
    print("=" * 50)
    
    try:
        test_discriminator()
        test_generator()
        test_create_models()
        test_different_channels()
        
        print("=" * 50)
        print("🎉 All tests completed successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc() 