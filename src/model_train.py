# =================================================================
# src/model_train.py (Stage 2: Model Development)
# =================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import numpy as np
import os
import matplotlib.pyplot as plt

# Import necessary components from data_prep
from src.data_prep import get_dataloaders, CLASSES, IMAGE_SIZE 

# --- 1. Configuration ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Calculate absolute path relative to the project root for robustness
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'models', 'best_model.pth')
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")   # <-- FIXED
NUM_CLASSES = len(CLASSES)
NUM_EPOCHS = 10  # Start with 10-15 epochs, increase if needed

# --- Ensure output directories exist ---
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def setup_model(num_classes=NUM_CLASSES):
    """Loads ResNet-18 with transfer learning setup."""
    print("Setting up ResNet-18 model with ImageNet pre-trained weights...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Freeze initial layers for feature extraction
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Unfreeze the new layer
    for param in model.fc.parameters():
        param.requires_grad = True
        
    return model.to(DEVICE)

def evaluate_model(model, data_loader, criterion=None, display_metrics=True):
    """Evaluates the model and computes key metrics."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            
            if criterion:
                total_loss += criterion(outputs, labels).item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset) if criterion else 0.0
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    if display_metrics:
        print(f"  -> Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    
    return accuracy, avg_loss, precision, recall, cm, all_preds, all_labels

def train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS):
    """Main training loop."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001) 
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        val_acc, val_loss, _, _, _, _, _ = evaluate_model(model, val_loader, criterion, display_metrics=False)
        
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> Saved Best Model checkpoint with Val Acc: {best_acc:.4f}")

    return model, history

def plot_and_save_cm(cm, classes, filename='confusion_matrix.png'):
    """Plots and saves the confusion matrix."""
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(im)
    
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="left")
    ax.set_yticklabels(classes)
    ax.xaxis.set_ticks_position("bottom")

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="red")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    
    plot_path = os.path.join(RESULTS_DIR, filename)   # <-- FIXED
    plt.savefig(plot_path)
    print(f"Saved Confusion Matrix plot to {plot_path}")
    
def plot_and_save_history(history, filename='training_history.png'):
    """Plots and saves the training history (Loss & Accuracy)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss vs. Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(history['val_acc'], label='Validation Accuracy', color='green')
    ax2.set_title('Validation Accuracy vs. Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plot_path = os.path.join(RESULTS_DIR, filename)   # <-- FIXED
    plt.savefig(plot_path)
    print(f"Saved Training History plot to {plot_path}")
    
# --- Main Execution ---
if __name__ == '__main__':
    train_loader, val_loader, _, idx_to_class = get_dataloaders(test_split=0.0) 
    model = setup_model()
    
    print("\n--- Starting Model Training ---")
    trained_model, training_history = train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS)
    
    trained_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    
    print("\n--- FINAL MODEL EVALUATION (Validation Set) ---")
    acc, loss, prec, rec, cm, _, _ = evaluate_model(trained_model, val_loader, nn.CrossEntropyLoss())
    
    plot_and_save_cm(cm, classes=list(idx_to_class.values()))
    plot_and_save_history(training_history)
