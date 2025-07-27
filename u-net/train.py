import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets import CarvanaDataset
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import UNet

# hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 5
NUM_WORKERS = 2
IMAGE_HEIGHT = 160 # 1280 originally
IMAGE_WIDTH = 240 # 1918 originally
LOAD_MODEL = False
LOAD_MODEL_PATH = "checkpoints/my_checkpoint.pth.tar"
TRAIN_IMG_DIR = "data/train"
TRAIN_MASK_DIR = "data/train_masks"

def train_fn(loader, model, optimizer, loss_fn, scaler, writer, epoch, global_step):
    model.train()
    loop = tqdm(loader)
    running_loss = 0.0
    running_accuracy = 0.0
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(DEVICE, dtype=torch.float32)
        y = y.to(DEVICE, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
        with torch.cuda.amp.autocast():
            preds = model(x)
            loss = loss_fn(preds, y)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        
        # Calculate accuracy for this batch
        preds_binary = torch.sigmoid(preds) > 0.5
        accuracy = (preds_binary == y).float().mean().item() * 100
        running_accuracy += accuracy
        
        loop.set_postfix(loss=loss.item(), accuracy=accuracy)
        writer.add_scalar("Loss/Train_batch", loss.item(), global_step)
        writer.add_scalar("Accuracy/Train_batch", accuracy, global_step)
        global_step += 1
    
    avg_loss = running_loss / len(loader)
    avg_accuracy = running_accuracy / len(loader)
    writer.add_scalar("Loss/Train_epoch", avg_loss, epoch)
    writer.add_scalar("Accuracy/Train_epoch", avg_accuracy, epoch)
    return global_step

def check_accuracy(loader, model, writer, epoch, global_step):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    iou_score = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(DEVICE, dtype=torch.float32)
            y = y.to(DEVICE, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice = (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
            dice_score += dice
            
            # Calculate IoU (Intersection over Union)
            intersection = (preds * y).sum()
            union = (preds + y).sum() - intersection
            iou = intersection / (union + 1e-8)
            iou_score += iou
            
            accuracy = (preds == y).sum() / torch.numel(preds) * 100
            writer.add_scalar("Accuracy/Val_batch", accuracy, global_step)
            writer.add_scalar("Dice/Val_batch", dice, global_step)
            writer.add_scalar("IoU/Val_batch", iou, global_step)
            
            # Log random validation images (first batch of each epoch)
            if batch_idx == 0:
                # Take first 4 images from the batch
                for i in range(min(4, x.size(0))):
                    # Input image (normalize to [0,1] for display)
                    input_img = (x[i] - x[i].min()) / (x[i].max() - x[i].min())
                    writer.add_image(f"Validation/Input_{i}", input_img, epoch)
                    
                    # Ground truth mask (convert to RGB for better visualization)
                    gt_mask = y[i].repeat(3, 1, 1)  # [3, H, W] - grayscale to RGB
                    writer.add_image(f"Validation/Ground_Truth_{i}", gt_mask, epoch)
                    
                    # Predicted mask (convert to RGB)
                    pred_mask = preds[i].repeat(3, 1, 1)  # [3, H, W] - grayscale to RGB
                    writer.add_image(f"Validation/Prediction_{i}", pred_mask, epoch)
                    
                    # Color-coded overlay: Green for correct predictions, Red for false positives, Blue for false negatives
                    overlay = input_img.clone()
                    
                    # True positives (green)
                    tp = (pred_mask[0] == 1) & (gt_mask[0] == 1)
                    overlay[0][tp] = 0  # Remove red
                    overlay[1][tp] = 1  # Add green
                    overlay[2][tp] = 0  # Remove blue
                    
                    # False positives (red)
                    fp = (pred_mask[0] == 1) & (gt_mask[0] == 0)
                    overlay[0][fp] = 1  # Add red
                    overlay[1][fp] = 0  # Remove green
                    overlay[2][fp] = 0  # Remove blue
                    
                    # False negatives (blue)
                    fn = (pred_mask[0] == 0) & (gt_mask[0] == 1)
                    overlay[0][fn] = 0  # Remove red
                    overlay[1][fn] = 0  # Remove green
                    overlay[2][fn] = 1  # Add blue
                    
                    writer.add_image(f"Validation/Segmentation_Overlay_{i}", overlay, epoch)
                    
                    # Dice score for this specific image
                    dice_img = (2 * (pred_mask[0] * gt_mask[0]).sum()) / ((pred_mask[0] + gt_mask[0]).sum() + 1e-8)
                    writer.add_scalar(f"Validation/Dice_Image_{i}", dice_img, epoch)
    
    accuracy = num_correct / num_pixels * 100
    avg_dice = dice_score / len(loader)
    avg_iou = iou_score / len(loader)
    print(f"Got {num_correct}/{num_pixels} with accuracy of {accuracy:.2f}")
    print(f"Dice score: {avg_dice:.4f}")
    print(f"IoU score: {avg_iou:.4f}")
    writer.add_scalar("Accuracy/Val_epoch", accuracy, epoch)
    writer.add_scalar("Dice/Val_epoch", avg_dice, epoch)
    writer.add_scalar("IoU/Val_epoch", avg_iou, epoch)
    model.train()

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            ToTensorV2(),
        ]
    )

    train_dataset = CarvanaDataset(
        image_dir=TRAIN_IMG_DIR,
        mask_dir=TRAIN_MASK_DIR,
        transform=train_transform,
    )

    # Suppose train_dataset is your full dataset
    total_size = len(train_dataset)
    val_size = int(0.05 * total_size)
    train_size = total_size - val_size

    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter(f"runs/UNet")

    global_step = 0
    for epoch in range(NUM_EPOCHS):
        global_step = train_fn(train_loader, model, optimizer, loss_fn, scaler, writer, epoch, global_step)
        check_accuracy(val_loader, model, writer, epoch, global_step)


if __name__ == "__main__":
    main()