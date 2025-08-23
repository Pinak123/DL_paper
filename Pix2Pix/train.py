import os
import argparse
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
from tqdm import tqdm

from model import UNetGenerator, PatchDiscriminator
from dataset import Pix2PixDataset


class Pix2PixTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.generator = UNetGenerator(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"]
        ).to(self.device)
        
        self.discriminator = PatchDiscriminator(
            in_channels=config["in_channels"]
        ).to(self.device)
        
        # Loss functions
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        
        # Optimizers
        self.gen_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=config["lr"],
            betas=(config["beta1"], 0.999)
        )
        self.disc_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=config["lr"],
            betas=(config["beta1"], 0.999)
        )
        
        # Learning rate schedulers
        self.gen_scheduler = optim.lr_scheduler.LinearLR(
            self.gen_optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=config["epochs"]
        )
        self.disc_scheduler = optim.lr_scheduler.LinearLR(
            self.disc_optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=config["epochs"]
        )
        
        # Tensorboard
        self.writer = SummaryWriter(config["log_dir"])
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.global_step = 0
        self.start_epoch = 0
        
        # Load checkpoint if resuming
        if config.get("resume_checkpoint"):
            self.load_checkpoint(config["resume_checkpoint"])
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "gen_optimizer_state_dict": self.gen_optimizer.state_dict(),
            "disc_optimizer_state_dict": self.disc_optimizer.state_dict(),
            "gen_scheduler_state_dict": self.gen_scheduler.state_dict(),
            "disc_scheduler_state_dict": self.disc_scheduler.state_dict(),
            "config": self.config,
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pth"
            torch.save(checkpoint, best_path)
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, latest_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.gen_optimizer.load_state_dict(checkpoint["gen_optimizer_state_dict"])
        self.disc_optimizer.load_state_dict(checkpoint["disc_optimizer_state_dict"])
        self.gen_scheduler.load_state_dict(checkpoint["gen_scheduler_state_dict"])
        self.disc_scheduler.load_state_dict(checkpoint["disc_scheduler_state_dict"])
        
        self.start_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Resuming from epoch {self.start_epoch}")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        real_A = batch["A"].to(self.device)  # condition
        real_B = batch["B"].to(self.device)  # target
        batch_size = real_A.size(0)
        
        # Labels for adversarial loss
        real_label = torch.ones(batch_size, 1, 30, 30, device=self.device)
        fake_label = torch.zeros(batch_size, 1, 30, 30, device=self.device)
        
        # Train Discriminator
        self.disc_optimizer.zero_grad()
        
        # Real loss
        disc_real = self.discriminator(real_A, real_B)
        disc_real_loss = self.adversarial_loss(disc_real, real_label)
        
        # Fake loss
        fake_B = self.generator(real_A)
        disc_fake = self.discriminator(real_A, fake_B.detach())
        disc_fake_loss = self.adversarial_loss(disc_fake, fake_label)
        
        # Total discriminator loss
        disc_loss = (disc_real_loss + disc_fake_loss) * 0.5
        disc_loss.backward()
        self.disc_optimizer.step()
        
        # Train Generator
        self.gen_optimizer.zero_grad()
        
        # Adversarial loss
        disc_fake = self.discriminator(real_A, fake_B)
        gen_adv_loss = self.adversarial_loss(disc_fake, real_label)
        
        # L1 loss
        gen_l1_loss = self.l1_loss(fake_B, real_B)
        
        # Total generator loss
        gen_loss = gen_adv_loss + self.config["lambda_l1"] * gen_l1_loss
        gen_loss.backward()
        self.gen_optimizer.step()
        
        return {
            "disc_loss": disc_loss.item(),
            "disc_real_loss": disc_real_loss.item(),
            "disc_fake_loss": disc_fake_loss.item(),
            "gen_loss": gen_loss.item(),
            "gen_adv_loss": gen_adv_loss.item(),
            "gen_l1_loss": gen_l1_loss.item(),
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.generator.eval()
        self.discriminator.eval()
        
        total_gen_loss = 0.0
        total_disc_loss = 0.0
        total_l1_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                real_A = batch["A"].to(self.device)
                real_B = batch["B"].to(self.device)
                batch_size = real_A.size(0)
                
                # Generate fake images
                fake_B = self.generator(real_A)
                
                # L1 loss
                l1_loss = self.l1_loss(fake_B, real_B)
                total_l1_loss += l1_loss.item()
                
                # Labels
                real_label = torch.ones(batch_size, 1, 30, 30, device=self.device)
                fake_label = torch.zeros(batch_size, 1, 30, 30, device=self.device)
                
                # Discriminator loss
                disc_real = self.discriminator(real_A, real_B)
                disc_fake = self.discriminator(real_A, fake_B)
                disc_loss = (
                    self.adversarial_loss(disc_real, real_label) +
                    self.adversarial_loss(disc_fake, fake_label)
                ) * 0.5
                total_disc_loss += disc_loss.item()
                
                # Generator loss
                gen_adv_loss = self.adversarial_loss(disc_fake, real_label)
                gen_loss = gen_adv_loss + self.config["lambda_l1"] * l1_loss
                total_gen_loss += gen_loss.item()
        
        num_batches = len(val_loader)
        return {
            "val_gen_loss": total_gen_loss / num_batches,
            "val_disc_loss": total_disc_loss / num_batches,
            "val_l1_loss": total_l1_loss / num_batches,
        }
    
    def log_images(self, batch: Dict[str, torch.Tensor], epoch: int):
        """Log sample images to tensorboard"""
        self.generator.eval()
        with torch.no_grad():
            real_A = batch["A"][:4].to(self.device)  # condition
            real_B = batch["B"][:4].to(self.device)  # target
            fake_B = self.generator(real_A)  # generated
            
            # Denormalize for visualization (from [-1,1] to [0,1])
            real_A = (real_A + 1) / 2
            real_B = (real_B + 1) / 2
            fake_B = (fake_B + 1) / 2
            
            # Clamp to valid range
            real_A = torch.clamp(real_A, 0, 1)
            real_B = torch.clamp(real_B, 0, 1)
            fake_B = torch.clamp(fake_B, 0, 1)
            
            # Log to tensorboard
            self.writer.add_images("Real_A", real_A, epoch)
            self.writer.add_images("Real_B", real_B, epoch)
            self.writer.add_images("Fake_B", fake_B, epoch)
        
        self.generator.train()
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        print(f"Starting training on {self.device}")
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
        
        best_val_loss = float("inf")
        
        for epoch in range(self.start_epoch, self.config["epochs"]):
            self.generator.train()
            self.discriminator.train()
            
            epoch_losses = {
                "disc_loss": 0.0,
                "gen_loss": 0.0,
                "gen_l1_loss": 0.0,
            }
            
            # Training loop
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
            for batch_idx, batch in enumerate(pbar):
                losses = self.train_step(batch)
                
                # Accumulate losses
                for key in epoch_losses:
                    if key in losses:
                        epoch_losses[key] += losses[key]
                
                # Log to tensorboard
                if self.global_step % self.config["log_interval"] == 0:
                    for key, value in losses.items():
                        self.writer.add_scalar(f"Train/{key}", value, self.global_step)
                    
                    # Log learning rates
                    self.writer.add_scalar("Learning_Rate/Generator", 
                                         self.gen_optimizer.param_groups[0]["lr"], self.global_step)
                    self.writer.add_scalar("Learning_Rate/Discriminator", 
                                         self.disc_optimizer.param_groups[0]["lr"], self.global_step)
                
                # Update progress bar
                pbar.set_postfix({
                    "D_loss": f"{losses['disc_loss']:.4f}",
                    "G_loss": f"{losses['gen_loss']:.4f}",
                    "L1": f"{losses['gen_l1_loss']:.4f}",
                })
                
                self.global_step += 1
            
            # Average epoch losses
            for key in epoch_losses:
                epoch_losses[key] /= len(train_loader)
            
            # Validation
            val_losses = self.validate(val_loader)
            
            # Log epoch losses
            for key, value in epoch_losses.items():
                self.writer.add_scalar(f"Epoch_Train/{key}", value, epoch)
            for key, value in val_losses.items():
                self.writer.add_scalar(f"Epoch_Val/{key}", value, epoch)
            
            # Log sample images
            if epoch % self.config["image_log_interval"] == 0:
                sample_batch = next(iter(val_loader))
                self.log_images(sample_batch, epoch)
            
            # Update learning rates
            self.gen_scheduler.step()
            self.disc_scheduler.step()
            
            # Save checkpoint
            is_best = val_losses["val_gen_loss"] < best_val_loss
            if is_best:
                best_val_loss = val_losses["val_gen_loss"]
            
            if (epoch + 1) % self.config["save_interval"] == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            print(f"Epoch {epoch+1}: "
                  f"G_loss={epoch_losses['gen_loss']:.4f}, "
                  f"D_loss={epoch_losses['disc_loss']:.4f}, "
                  f"Val_G_loss={val_losses['val_gen_loss']:.4f}")
        
        self.writer.close()
        print("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train Pix2Pix model")
    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset")
    parser.add_argument("--log_dir", type=str, default="logs", help="Tensorboard log directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume_checkpoint", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--beta1", type=float, default=0.5, help="Beta1 for Adam optimizer")
    parser.add_argument("--lambda_l1", type=float, default=100.0, help="L1 loss weight")
    parser.add_argument("--image_size", type=int, default=256, help="Image size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval (steps)")
    parser.add_argument("--image_log_interval", type=int, default=5, help="Image logging interval (epochs)")
    parser.add_argument("--save_interval", type=int, default=10, help="Checkpoint save interval (epochs)")
    
    args = parser.parse_args()
    
    config = {
        "data_root": args.data_root,
        "log_dir": args.log_dir,
        "checkpoint_dir": args.checkpoint_dir,
        "resume_checkpoint": args.resume_checkpoint,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "beta1": args.beta1,
        "lambda_l1": args.lambda_l1,
        "image_size": args.image_size,
        "num_workers": args.num_workers,
        "log_interval": args.log_interval,
        "image_log_interval": args.image_log_interval,
        "save_interval": args.save_interval,
        "in_channels": 3,
        "out_channels": 3,
    }
    
    # Create datasets
    train_dataset = Pix2PixDataset(
        root=config["data_root"],
        mode="aligned",
        split="train",
        image_size=config["image_size"]
    )
    
    val_dataset = Pix2PixDataset(
        root=config["data_root"],
        mode="aligned",
        split="val",
        image_size=config["image_size"],
        random_jitter=False  # No augmentation for validation
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    
    # Initialize trainer
    trainer = Pix2PixTrainer(config)
    
    # Start training
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
