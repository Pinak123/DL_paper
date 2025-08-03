import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import os
import time
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

from model import UNet
from data_loader import create_data_loaders

class DiceLoss(nn.Module):
    """Dice loss for medical segmentation"""
    
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice

def dice_coefficient(pred, target, smooth=1e-6):
    """Calculate Dice coefficient"""
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return dice

class LGGTrainer:
    """Trainer class for LGG MRI segmentation"""
    
    def __init__(self, model, train_loader, val_loader, device, lr=1e-4, log_dir="runs/lgg_training"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss and optimizer - use BCE + Dice for better training
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_dice_scores = []
        
        # Best model tracking
        self.best_val_dice = 0.0
        
        # TensorBoard setup - simplified
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (images, masks) in enumerate(pbar):
            images, masks = images.to(self.device), masks.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            bce_loss = self.bce_loss(outputs, masks)
            dice_loss = self.dice_loss(outputs, masks)
            loss = bce_loss + dice_loss  # Combined loss
            dice = dice_coefficient(outputs, masks)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_dice += dice.item()
            
            # Simplified TensorBoard logging - only essentials
            if self.global_step % 50 == 0:  # Log every 50 batches instead of every batch
                self.writer.add_scalar('Loss/Training', loss.item(), self.global_step)
                self.writer.add_scalar('Dice/Training', dice.item(), self.global_step)
                
                # Log images every 200 batches
                if self.global_step % 200 == 0:
                    self._log_images(images, masks, outputs, 'Training', self.global_step)
            
            self.global_step += 1
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Dice': f'{dice.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        avg_dice = total_dice / len(self.train_loader)
        
        # Log epoch-level metrics
        self.writer.add_scalar('Loss/TrainingEpoch', avg_loss, epoch)
        self.writer.add_scalar('Dice/TrainingEpoch', avg_dice, epoch)
        
        return avg_loss
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for batch_idx, (images, masks) in enumerate(pbar):
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                
                bce_loss = self.bce_loss(outputs, masks)
                dice_loss = self.dice_loss(outputs, masks)
                loss = bce_loss + dice_loss  # Combined loss
                dice = dice_coefficient(outputs, masks)
                
                total_loss += loss.item()
                total_dice += dice.item()
                
                # Simplified validation logging - only images
                if batch_idx == 0:  # Log images from first batch only
                    self._log_images(images, masks, outputs, 'Validation', epoch)
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}', 
                    'Dice': f'{dice.item():.4f}'
                })
        
        avg_loss = total_loss / len(self.val_loader)
        avg_dice = total_dice / len(self.val_loader)
        
        # Log epoch-level validation metrics
        self.writer.add_scalar('Loss/ValidationEpoch', avg_loss, epoch)
        self.writer.add_scalar('Dice/ValidationEpoch', avg_dice, epoch)
        
        return avg_loss, avg_dice
    
    def _log_images(self, images, masks, outputs, phase, step):
        """Log images to TensorBoard - simplified"""
        # Take first image from batch
        image = images[0].cpu()
        mask = masks[0].cpu()
        output = outputs[0].cpu()  # Model outputs probabilities directly
        
        # Normalize images for display
        image_norm = (image - image.min()) / (image.max() - image.min())
        
        # Log only the essential comparison: Original vs Ground Truth vs Prediction
        self.writer.add_image(f'{phase}/Original_Image', image_norm, step)
        self.writer.add_image(f'{phase}/Ground_Truth_Mask', mask, step)
        self.writer.add_image(f'{phase}/Predicted_Mask', output, step)
    
    def train(self, num_epochs=50):
        """Train the model"""
        print(f"🚀 Starting training for {num_epochs} epochs...")
        print(f"📊 Training samples: {len(self.train_loader.dataset)}")
        print(f"📊 Validation samples: {len(self.val_loader.dataset)}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
            
            # Training
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss, val_dice = self.validate_epoch(epoch)
            self.val_losses.append(val_loss)
            self.val_dice_scores.append(val_dice)
            
            # Learning rate scheduling
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            
            # Log learning rate changes
            if old_lr != new_lr:
                print(f"📉 Learning rate reduced from {old_lr:.2e} to {new_lr:.2e}")
            
            # Save best model
            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_dice': val_dice,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, 'best_lgg_model.pth')
                print(f"✅ New best model saved! Dice: {val_dice:.4f}")
            
            print(f"📈 Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        total_time = time.time() - start_time
        print(f"⏱️  Total training time: {total_time/3600:.2f} hours")
        
        # Minimal final summary
        print(f"📋 Training Summary:")
        print(f"   Best Validation Dice: {self.best_val_dice:.4f}")
        print(f"   Final Training Loss: {self.train_losses[-1]:.4f}")
        print(f"   Final Validation Loss: {self.val_losses[-1]:.4f}")
        
        return self.train_losses, self.val_losses, self.val_dice_scores
    
    def close_writer(self):
        """Close TensorBoard writer"""
        self.writer.close()
    
    def plot_training_results(self):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Val Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot Dice scores
        ax2.plot(self.val_dice_scores, label='Val Dice Score', color='green')
        ax2.set_title('Validation Dice Score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Dice Score')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('lgg_training_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main training function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Create data loaders
    data_dir = "data/lgg-mri-segmentation/kaggle_3m"
    train_loader, val_loader = create_data_loaders(data_dir, batch_size=4)
    
    # Initialize model
    model = UNet(in_channels=1, out_channels=1).to(device)
    print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = LGGTrainer(model, train_loader, val_loader, device, lr=1e-4)
    
    # Train model
    train_losses, val_losses, val_dice_scores = trainer.train(num_epochs=30)
    
    # Plot results
    trainer.plot_training_results()
    
    # Close TensorBoard writer
    trainer.close_writer()
    
    print("✅ Training completed!")
    print(f"🏆 Best validation Dice score: {max(val_dice_scores):.4f}")
    print(f"📊 TensorBoard logs saved in: runs/lgg_training")
    print(f"🚀 To view TensorBoard, run: tensorboard --logdir=runs")

if __name__ == "__main__":
    main() 