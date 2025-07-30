#!/usr/bin/env python3
"""
DCGAN Training Runner
This script runs the DCGAN training with proper setup and error handling.
"""

import sys
import os
import torch

def check_requirements():
    """Check if all required packages are available"""
    try:
        import torch
        import torchvision
        import matplotlib
        import tqdm
        print("✅ All required packages are available")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please install required packages: pip install torch torchvision matplotlib tqdm tensorboard")
        return False

def check_gpu():
    """Check GPU availability"""
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("⚠️  No GPU detected, using CPU (training will be slower)")
        return False

def main():
    """Main training function"""
    print("=" * 60)
    print("DCGAN Training Setup")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check GPU
    check_gpu()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create necessary directories
    os.makedirs("dataset", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("generated_images", exist_ok=True)
    
    print("\n" + "=" * 60)
    print("Starting DCGAN Training")
    print("=" * 60)
    
    try:
        # Import and run training
        import train
        print("Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 