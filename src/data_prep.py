# src/data_prep.py (Stage 1: Dataset Preparation)
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

# --- 1. Configuration ---
IMAGE_SIZE = 224
BATCH_SIZE = 32
# Define DATA_DIR relative to the project root
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'trashnet') 
# TrashNet classes: Cardboard, Glass, Metal, Paper, Plastic, Trash
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Standard ImageNet normalization values (good for transfer learning)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# --- 2. Define Preprocessing and Augmentation ---

# Training Transformations (includes augmentation for robustness)
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE), # Random cropping and resizing
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),           # Augmentation
    transforms.ColorJitter(brightness=0.1, contrast=0.1), # Augmentation
    transforms.ToTensor(),                   # Convert to PyTorch tensor
    transforms.Normalize(NORM_MEAN, NORM_STD) # Normalization
])

# Validation/Test/Inference Preprocessing (Deterministic, NO augmentation)
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE), # Crop center to get 224x224
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD)
])

def get_dataloaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE, val_split=0.2, test_split=0.1):
    """
    Loads data using ImageFolder, applies training transforms, and splits the data.
    """
    # Load all data using ImageFolder (applies train_transforms initially)
    full_dataset = datasets.ImageFolder(data_dir, transform=train_transforms) 
    
    # Calculate split sizes
    total_size = len(full_dataset)
    test_size = int(test_split * total_size)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size - test_size
    
    # Randomly split the dataset indices
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42) # Set seed for reproducibility
    )
    
    # IMPORTANT: Apply correct (non-augmented) transformations to val and test sets
    # We must explicitly set the transform function for the Subset objects
    val_dataset.dataset.transform = test_transforms 
    test_dataset.dataset.transform = test_transforms

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    # Batch size 1 for the test loader, as it will be used to simulate single-frame input
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4) 
    
    # Extract class mapping for reporting and inference
    class_to_idx = full_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    print(f"Dataset Loaded: {total_size} images. Split: Train={train_size}, Validation={val_size}, Test={test_size}")
    return train_loader, val_loader, test_loader, idx_to_class

# A simple check that the script is executable and data can be loaded
if __name__ == '__main__':
    try:
        train, val, test, classes = get_dataloaders()
        print(f"Class Mapping: {classes}")
        print("Data preparation complete and verified.")
    except Exception as e:
        print(f"Error loading data. Check your file path: {DATA_DIR}. Error: {e}")