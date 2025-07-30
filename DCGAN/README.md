# DCGAN Training Implementation

This directory contains a complete DCGAN (Deep Convolutional Generative Adversarial Network) implementation with training, visualization, and monitoring capabilities.

## Files Overview

- `model.py` - DCGAN Generator and Discriminator architecture
- `train.py` - Complete training loop with TensorBoard logging
- `test_models.py` - Model testing and validation script
- `run_training.py` - Training runner with error handling
- `README.md` - This file

## Features

✅ **Complete DCGAN Architecture** - Following the original paper specifications
✅ **LeakyReLU Activations** - For better training stability
✅ **TensorBoard Logging** - Real-time monitoring of losses and generated images
✅ **Image Visualization** - Generated images saved every 100 batches
✅ **Progress Bars** - Real-time training progress with metrics
✅ **Model Checkpoints** - Automatic model saving every 10 epochs
✅ **Label Smoothing** - Improved training stability
✅ **GPU Support** - Automatic GPU detection and utilization

## Installation

```bash
pip install torch torchvision matplotlib tqdm tensorboard
```

## Usage

### Quick Start

```bash
python run_training.py
```

### Manual Training

```bash
python train.py
```

### Test Models

```bash
python test_models.py
```

## Training Configuration

The training script uses the following hyperparameters (modifiable in `train.py`):

- **Learning Rate**: 2e-4
- **Batch Size**: 128
- **Image Size**: 28x28 (MNIST)
- **Channels**: 1 (grayscale)
- **Z Dimension**: 100
- **Epochs**: 50
- **Optimizer**: Adam with β1=0.5, β2=0.999

## Monitoring Training

### TensorBoard

Start TensorBoard to monitor training:

```bash
tensorboard --logdir=logs
```

Then open http://localhost:6006 in your browser.

### Generated Images

Images are saved in the `generated_images/` directory:
- `epoch_X_batch_Y.png` - Generated images at specific epochs/batches
- Real vs. fake image comparisons in TensorBoard

### Model Checkpoints

Models are automatically saved:
- `generator_epoch_X.pth` - Generator checkpoints every 10 epochs
- `discriminator_epoch_X.pth` - Discriminator checkpoints every 10 epochs
- `generator_final.pth` - Final generator model
- `discriminator_final.pth` - Final discriminator model

## Architecture Details

### Generator
- Input: Random noise (100-dimensional)
- Output: 28x28 grayscale images (MNIST)
- Architecture: 4 transposed convolutional layers (100→512→256→128→1)
- Activation: ReLU + Tanh output

### Discriminator
- Input: 28x28 grayscale images (MNIST)
- Output: Binary classification (real/fake)
- Architecture: 4 convolutional layers (1→64→128→256→1)
- Activation: LeakyReLU (0.2) + Sigmoid output

## Training Tips

1. **Monitor Losses**: Generator and discriminator losses should be balanced
2. **Check Images**: Generated images should improve over time
3. **Real vs Fake Scores**: Should converge to around 0.5 each
4. **GPU Memory**: Reduce batch size if you run out of memory

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `BATCH_SIZE` in `train.py`
2. **Training Instability**: The script includes label smoothing to help
3. **Poor Image Quality**: Try training for more epochs or adjust learning rate

### Performance Tips

- Use GPU for faster training
- Monitor TensorBoard for real-time metrics
- Check generated images periodically
- Save models regularly for recovery

## Expected Output

During training, you should see:
- Progress bars with real-time metrics
- Generated images improving over time
- TensorBoard logs with losses and images
- Model checkpoints saved automatically

The training will generate realistic-looking images from random noise, demonstrating the power of GANs for image generation. 