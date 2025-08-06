import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import os
from tqdm import tqdm

from model import Discriminator, Generator
from config import *
from utils import save_checkpoint, load_checkpoint, save_image
from datasets import HorseZebraDataset


def train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler, writer, epoch, global_step):
    """
    Train function for one epoch
    
    Args:
        disc_H, disc_Z: Horse and Zebra discriminators
        gen_Z, gen_H: Zebra and Horse generators
        loader: DataLoader
        opt_disc, opt_gen: Discriminator and Generator optimizers
        L1, mse: Loss functions
        d_scaler, g_scaler: Mixed precision scalers
        writer: TensorBoard writer
        epoch: Current epoch number
        global_step: Global step counter
    
    Returns:
        Updated global_step counter
    """
    loop = tqdm(loader, leave=True)
    
    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(DEVICE)
        horse = horse.to(DEVICE)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_horse = gen_H(zebra)
            D_H_real = disc_H(horse)
            D_H_fake = disc_H(fake_horse.detach())
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            fake_zebra = gen_Z(horse)
            D_Z_real = disc_Z(zebra)
            D_Z_fake = disc_Z(fake_zebra.detach())
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # Put it together
            D_loss = (D_H_loss + D_Z_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # Adversarial loss for both generators
            D_H_fake = disc_H(fake_horse)
            D_Z_fake = disc_Z(fake_zebra)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # Cycle loss
            cycle_zebra = gen_Z(fake_horse)
            cycle_horse = gen_H(fake_zebra)
            cycle_zebra_loss = L1(zebra, cycle_zebra)
            cycle_horse_loss = L1(horse, cycle_horse)

            # Identity loss (remove for efficiency if you want)
            identity_zebra = gen_Z(zebra)
            identity_horse = gen_H(horse)
            identity_zebra_loss = L1(zebra, identity_zebra)
            identity_horse_loss = L1(horse, identity_horse)

            # Add all losses together
            G_loss = (
                loss_G_Z
                + loss_G_H
                + cycle_zebra_loss * LAMBDA_CYCLE
                + cycle_horse_loss * LAMBDA_CYCLE
                + identity_horse_loss * LAMBDA_IDENTITY
                + identity_zebra_loss * LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        # Log losses to TensorBoard
        if idx % 10 == 0:
            writer.add_scalar('Loss/Discriminator_H', D_H_loss.item(), global_step)
            writer.add_scalar('Loss/Discriminator_Z', D_Z_loss.item(), global_step)
            writer.add_scalar('Loss/Generator_Total', G_loss.item(), global_step)
            writer.add_scalar('Loss/Generator_H', loss_G_H.item(), global_step)
            writer.add_scalar('Loss/Generator_Z', loss_G_Z.item(), global_step)
            writer.add_scalar('Loss/Cycle_Zebra', cycle_zebra_loss.item(), global_step)
            writer.add_scalar('Loss/Cycle_Horse', cycle_horse_loss.item(), global_step)
            writer.add_scalar('Loss/Identity_Zebra', identity_zebra_loss.item(), global_step)
            writer.add_scalar('Loss/Identity_Horse', identity_horse_loss.item(), global_step)

        # Save sample images and log to TensorBoard
        if idx % 200 == 0 and idx > 0:
            with torch.no_grad():
                # Denormalize images for saving (from [-1,1] to [0,1])
                fake_horse_denorm = fake_horse * 0.5 + 0.5
                fake_zebra_denorm = fake_zebra * 0.5 + 0.5
                cycle_zebra_denorm = cycle_zebra * 0.5 + 0.5
                cycle_horse_denorm = cycle_horse * 0.5 + 0.5
                real_horse_denorm = horse * 0.5 + 0.5
                real_zebra_denorm = zebra * 0.5 + 0.5

                # Save individual images
                save_image(fake_horse_denorm, f"saved_images/fake_horse_epoch_{epoch}_step_{idx}.png")
                save_image(fake_zebra_denorm, f"saved_images/fake_zebra_epoch_{epoch}_step_{idx}.png")
                save_image(cycle_zebra_denorm, f"saved_images/cycle_zebra_epoch_{epoch}_step_{idx}.png")
                save_image(cycle_horse_denorm, f"saved_images/cycle_horse_epoch_{epoch}_step_{idx}.png")

                # Log images to TensorBoard
                img_grid_real = vutils.make_grid([real_zebra_denorm[0], real_horse_denorm[0]], normalize=False)
                img_grid_fake = vutils.make_grid([fake_zebra_denorm[0], fake_horse_denorm[0]], normalize=False)
                img_grid_cycle = vutils.make_grid([cycle_zebra_denorm[0], cycle_horse_denorm[0]], normalize=False)

                writer.add_image('Images/Real_Zebra_Horse', img_grid_real, global_step)
                writer.add_image('Images/Fake_Zebra_Horse', img_grid_fake, global_step)
                writer.add_image('Images/Cycle_Zebra_Horse', img_grid_cycle, global_step)

        # Update progress bar
        if idx % 50 == 0:
            loop.set_postfix(
                D_loss=D_loss.item(),
                G_loss=G_loss.item(),
                D_H_real=torch.sigmoid(D_H_real).mean().item(),
                D_H_fake=torch.sigmoid(D_H_fake).mean().item(),
            )

        global_step += 1
    
    return global_step


def train_model():
    # Create TensorBoard writer
    writer = SummaryWriter(log_dir='logs')
    
    # Dataset and DataLoader
    dataset = HorseZebraDataset(root_zebra=TRAIN_DIR_B, root_horse=TRAIN_DIR_A, transform=transforms)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    # Models
    gen_H = Generator(img_channels=3, num_features=64, num_residuals=9).to(DEVICE)  # Horse generator (Z->H)
    gen_Z = Generator(img_channels=3, num_features=64, num_residuals=9).to(DEVICE)  # Zebra generator (H->Z)
    disc_H = Discriminator(in_channels=3, features=[64, 128, 256, 512]).to(DEVICE)  # Horse discriminator
    disc_Z = Discriminator(in_channels=3, features=[64, 128, 256, 512]).to(DEVICE)  # Zebra discriminator

    # Optimizers
    opt_gen = optim.Adam(
        list(gen_H.parameters()) + list(gen_Z.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999)
    )
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    # Loss functions
    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    # Load checkpoints if specified
    if LOAD_MODEL:
        load_checkpoint(CHECKPOINT_GEN_H, gen_H, opt_gen, LEARNING_RATE)
        load_checkpoint(CHECKPOINT_GEN_Z, gen_Z, opt_gen, LEARNING_RATE)
        load_checkpoint(CHECKPOINT_CRITIC_H, disc_H, opt_disc, LEARNING_RATE)
        load_checkpoint(CHECKPOINT_CRITIC_Z, disc_Z, opt_disc, LEARNING_RATE)

    # Mixed precision training
    g_scaler = torch.amp.GradScaler('cuda')
    d_scaler = torch.amp.GradScaler('cuda')

    global_step = 0
    
    for epoch in range(NUM_EPOCHS):
        global_step = train_fn(
            disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, 
            L1, mse, d_scaler, g_scaler, writer, epoch, global_step
        )

        # Save checkpoints at the end of each epoch
        if SAVE_MODEL:
            save_checkpoint(gen_H, opt_gen, filename=CHECKPOINT_GEN_H)
            save_checkpoint(gen_Z, opt_gen, filename=CHECKPOINT_GEN_Z)
            save_checkpoint(disc_H, opt_disc, filename=CHECKPOINT_CRITIC_H)
            save_checkpoint(disc_Z, opt_disc, filename=CHECKPOINT_CRITIC_Z)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] completed")

    writer.close()
    print("Training completed!")


if __name__ == "__main__":
    train_model()