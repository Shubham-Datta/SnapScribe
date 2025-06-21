# SnapScribe - Image Captioning with CNN-Transformer Architecture

This project implements an image captioning model using a CNN-Transformer architecture on the Flickr8k dataset. The model combines EfficientNetB0 for image feature extraction with a custom Transformer encoder-decoder architecture for caption generation.

## Overview

The model generates natural language descriptions for images by:
- Extracting visual features using a pre-trained EfficientNetB0 CNN
- Encoding these features using a Transformer encoder
- Generating captions word-by-word using a Transformer decoder with attention mechanisms

## Features

- **CNN Feature Extraction**: Uses EfficientNetB0 pre-trained on ImageNet
- **Transformer Architecture**: Custom implementation of encoder-decoder with multi-head attention
- **Data Augmentation**: Random flips, rotations, and contrast adjustments for robust training
- **Learning Rate Scheduling**: Warm-up schedule for stable training
- **Early Stopping**: Prevents overfitting with patience-based stopping

## Requirements

- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Google Colab (recommended for T4 GPU support)

## Dataset

The model uses the **Flickr8k dataset**, which contains:
- 8,000 images
- Diverse scenes and objects from everyday life

The dataset is automatically downloaded when running the notebook.

## Model Architecture

### 1. Image Encoder
- **Base Model**: EfficientNetB0 (frozen weights)
- **Input Size**: 299x299x3
- **Output**: Flattened feature vectors

### 2. Transformer Encoder
- Single-head attention mechanism
- Layer normalization
- Dense feed-forward network
- Embedding dimension: 512

### 3. Transformer Decoder
- Two-head multi-head attention
- Causal masking for autoregressive generation
- Positional embeddings
- Vocabulary size: 10,000 tokens
- Maximum sequence length: 25 tokens

## Training Configuration

- **Batch Size**: 64
- **Epochs**: 30 (with early stopping)
- **Optimizer**: Adam with learning rate scheduling
- **Loss Function**: Sparse Categorical Crossentropy
- **Train/Validation Split**: 80/20

## Usage

1. **Environment Setup**: The notebook automatically sets up the Keras backend and downloads required data.

2. **Run in Google Colab**: 
   - Open the notebook in Google Colab
   - Select Runtime → Change runtime type → T4 GPU
   - Run all cells sequentially

3. **Training**: The model trains automatically with progress displayed for each epoch.

4. **Inference**: After training, the `generate_caption()` function displays random validation images with generated captions.

## Model Performance

The model tracks two key metrics during training:
- **Loss**: Sparse categorical crossentropy
- **Accuracy**: Token-level prediction accuracy

Early stopping ensures the best model weights are restored based on validation performance.

## Key Components

### Data Preprocessing
- Filters captions by length (5 to 25 tokens)
- Adds `<start>` and `<end>` tokens
- Applies text standardization and vectorization

### Custom Layers
- `TransformerEncoderBlock`: Processes image features
- `TransformerDecoderBlock`: Generates captions with attention
- `PositionalEmbedding`: Adds position information to tokens
- `ImageCaptioningModel`: Orchestrates the full pipeline

### Training Strategy
- Processes multiple captions per image in each batch
- Implements teacher forcing during training
- Uses causal masking for proper sequence generation

## Example Output

The model generates captions like:
- "a dog is running through the grass"
- "a person riding a bike on a path"
- "two people sitting on a bench near water"

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

We gratefully acknowledge the following resources and tools that made this project possible:
-Flickr8k Dataset: Provided by the University of Illinois at Urbana-Champaign, offering a high-quality benchmark for image captioning research.

-EfficientNetB0: Pre-trained model by Google Brain, used for feature extraction from images.

-TensorFlow & Keras: For providing a powerful deep learning framework to build and train the model.

-Google Colab: For offering free access to GPUs and an excellent cloud-based environment for development and training.

