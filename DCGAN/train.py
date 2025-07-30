import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as utils
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import create_dcgan_models, weights_init
import os
import numpy as np
from datetime import datetime
from tqdm import tqdm

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4
BATCH_SIZE = 64  # Reduced for better stability
IMAGE_SIZE = 28  # MNIST original size
IMAGE_CHANNELS = 1
Z_DIM = 100
NUM_CLASSES = 10
NUM_EPOCHS = 100  # Increased epochs
FEAT_DISC = 64
FEAT_GEN = 64
CHANNELS_IMG = 1

# Print hyperparameters
print(f"Training Configuration:")
print(f"  Device: {device}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Learning Rate: {LEARNING_RATE}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Z Dimension: {Z_DIM}")

# Create directories
os.makedirs("dataset", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("generated_images", exist_ok=True)

# Data transforms
transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # MNIST is already grayscale
])

# Load dataset
dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True, train=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# Create models
gen, disc = create_dcgan_models(Z_DIM, IMAGE_SIZE, IMAGE_CHANNELS, NUM_CLASSES)
gen.to(device)
disc.to(device)

# Initialize weights
gen.apply(weights_init)
disc.apply(weights_init)

# Loss function
criterion = nn.BCELoss()

# Optimizers
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

# Learning rate schedulers for better convergence
scheduler_gen = optim.lr_scheduler.StepLR(opt_gen, step_size=30, gamma=0.5)
scheduler_disc = optim.lr_scheduler.StepLR(opt_disc, step_size=30, gamma=0.5)

# TensorBoard
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(f"logs/dcgan_{timestamp}")

# Training statistics
step = 0
fixed_noise = torch.randn(64, Z_DIM).to(device)  # Fixed noise for consistent visualization

# Add noise to labels for label smoothing (helps with training stability)
real_label = 0.9  # Instead of 1.0
fake_label = 0.0

# Training frequency
n_critic = 1  # Balance training - train both equally

def save_generated_images(images, epoch, batch_idx, step):
    """Save generated images to disk and TensorBoard"""
    # Denormalize images from [-1, 1] to [0, 1]
    images = (images + 1) / 2.0
    images = torch.clamp(images, 0, 1)
    
    # Save to disk
    grid = utils.make_grid(images, nrow=8, padding=2, normalize=False)
    # utils.save_image(grid, f"generated_images/epoch_{epoch}_batch_{batch_idx}.png")
    
    # Log to TensorBoard
    writer.add_image(f"Generated_Images/Epoch_{epoch}", grid, step)

def log_metrics(step, gen_loss, disc_loss, real_score, fake_score):
    """Log training metrics to TensorBoard"""
    writer.add_scalar("Loss/Generator", gen_loss, step)
    writer.add_scalar("Loss/Discriminator", disc_loss, step)
    writer.add_scalar("Scores/Real", real_score, step)
    writer.add_scalar("Scores/Fake", fake_score, step)

def train_discriminator(real_images, batch_size):
    """Train discriminator on real and fake images"""
    opt_disc.zero_grad()
    
    # Real images
    real_labels = torch.ones(batch_size, 1, 1, 1).to(device) * real_label
    real_output = disc(real_images)
    real_loss = criterion(real_output, real_labels)
    
    # Fake images
    noise = torch.randn(batch_size, Z_DIM).to(device)
    fake_images = gen(noise)
    fake_labels = torch.zeros(batch_size, 1, 1, 1).to(device)
    fake_output = disc(fake_images.detach())
    fake_loss = criterion(fake_output, fake_labels)
    
    # Total discriminator loss
    disc_loss = (real_loss + fake_loss) / 2
    disc_loss.backward()
    opt_disc.step()
    
    return disc_loss, real_output.mean().item(), fake_output.mean().item()

def train_generator(batch_size):
    """Train generator to fool discriminator"""
    opt_gen.zero_grad()
    
    noise = torch.randn(batch_size, Z_DIM).to(device)
    fake_images = gen(noise)
    fake_labels = torch.ones(batch_size, 1, 1, 1).to(device) * real_label  # Generator wants discriminator to think these are real
    fake_output = disc(fake_images)
    gen_loss = criterion(fake_output, fake_labels)
    
    gen_loss.backward()
    opt_gen.step()
    
    return gen_loss

# Training loop
print(f"Starting DCGAN training on {device}")
print(f"Dataset size: {len(dataset)}")
print(f"Number of batches per epoch: {len(loader)}")

for epoch in range(NUM_EPOCHS):
    gen.train()
    disc.train()
    
    epoch_gen_loss = 0
    epoch_disc_loss = 0
    
    # Create progress bar for this epoch
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    
    for batch_idx, (real_images, _) in enumerate(pbar):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        
        # Train discriminator
        disc_loss, real_score, fake_score = train_discriminator(real_images, batch_size)
        
        # Train generator every iteration for better balance
        gen_loss = train_generator(batch_size)
        
        # Log metrics
        if batch_idx % 10 == 0:
            log_metrics(step, gen_loss.item(), disc_loss.item(), real_score, fake_score)
            pbar.set_postfix({
                'Gen Loss': f'{gen_loss.item():.4f}',
                'Disc Loss': f'{disc_loss.item():.4f}',
                'Real Score': f'{real_score:.4f}',
                'Fake Score': f'{fake_score:.4f}'
            })
        
        # Generate and save images periodically
        if batch_idx % 50 == 0:  # More frequent visualization
            gen.eval()
            with torch.no_grad():
                fake_images = gen(fixed_noise)
                save_generated_images(fake_images, epoch, batch_idx, step)
                
                # Also log some real images for comparison
                real_grid = utils.make_grid(real_images[:16], nrow=4, padding=2, normalize=True)
                writer.add_image(f"Real_Images/Epoch_{epoch}", real_grid, step)
            gen.train()
        
        step += 1
        epoch_gen_loss += gen_loss.item()
        epoch_disc_loss += disc_loss.item()
    
    # Log epoch statistics
    avg_gen_loss = epoch_gen_loss / len(loader)
    avg_disc_loss = epoch_disc_loss / len(loader)
    writer.add_scalar("Loss/Avg_Generator_Epoch", avg_gen_loss, epoch)
    writer.add_scalar("Loss/Avg_Discriminator_Epoch", avg_disc_loss, epoch)
    
    # Step learning rate schedulers
    scheduler_gen.step()
    scheduler_disc.step()
    
    print(f"Epoch [{epoch}/{NUM_EPOCHS}] - Avg Gen Loss: {avg_gen_loss:.4f}, Avg Disc Loss: {avg_disc_loss:.4f}")
    
    # Save models periodically
    if (epoch + 1) % 10 == 0:
        torch.save(gen.state_dict(), f"generator_epoch_{epoch+1}.pth")
        torch.save(disc.state_dict(), f"discriminator_epoch_{epoch+1}.pth")

# Save final models
torch.save(gen.state_dict(), "generator_final.pth")
torch.save(disc.state_dict(), "discriminator_final.pth")

print("Training completed!")
writer.close()


