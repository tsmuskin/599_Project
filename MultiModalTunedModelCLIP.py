"""
CLIP-based Hate Speech Classifier
Modified to use a larger CLIP backbone (vit-large) and differential learning rates
by unfreezing the last few layers for fine-tuning.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from DataCleanerClass import MultimodalHateSpeechDataCleaner
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os 
from safetensors.torch import save_model, load_model

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Step 1: Load and clean data
BASE_PATH = '/projectnb/cs599m1/projects/multimodal-hatespeech/599_Project'
MODEL_SAVE_PATH = 'best_clip_classifier.safetensors' 

print("Processing data...")
cleaner = MultimodalHateSpeechDataCleaner(BASE_PATH, random_state=42)
cleaner.process_all(load_images=False, balance=True) 

# Get the data
data = cleaner.get_data()
X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['X_test']
y_test = data['y_test']

print(f"Train samples: {len(X_train)}")
print(f"Val samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# Step 2: Create Dataset (No changes needed here)
class CLIPHateSpeechDataset(Dataset):
    def __init__(self, X, y, processor, max_length=77):
        self.image_paths = X['image_path'].values
        self.texts = X['cleaned_text'].values
        self.labels = y['binary_label'].values
        self.processor = processor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Load image
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
        except Exception as e:
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), color='white')
        
        # Get text
        text = self.texts[idx]
        
        # Process with CLIP (using cleaned text - can be improved with prompting if desired)
        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Step 3: Define Model (Major changes here)
class CLIPHateSpeechClassifier(nn.Module):
    # use CLIP model vit large
    def __init__(self, clip_model_name="openai/clip-vit-large-patch14", unfreeze_layers=2):
        super().__init__()
        
        # Load CLIP, prioritizing safetensors to bypass the torch.load security check
        self.clip = CLIPModel.from_pretrained(
            clip_model_name,
            use_safetensors=True 
        )
        
        #  Unfreeze the last N layers of the Vision and Text encoders
        self.unfreeze_layers = unfreeze_layers
        
        # Get all parameters that are NOT the classification head
        backbone_params = []

        # --- Vision Encoder Unfreezing ---
        # The CLIP ViT has transformer blocks (layers). unfreeze the last N blocks.
        vision_layers = self.clip.vision_model.encoder.layers
        for i, layer in enumerate(vision_layers):
            if i >= len(vision_layers) - self.unfreeze_layers:
                for param in layer.parameters():
                    param.requires_grad = True
                    backbone_params.append(param)
            else:
                for param in layer.parameters():
                    param.requires_grad = False
        
        # --- Text Encoder Unfreezing ---
        text_layers = self.clip.text_model.encoder.layers
        for i, layer in enumerate(text_layers):
            if i >= len(text_layers) - self.unfreeze_layers:
                for param in layer.parameters():
                    param.requires_grad = True
                    backbone_params.append(param)
            else:
                for param in layer.parameters():
                    param.requires_grad = False
        
        # Get embedding dimension (768 for large-patch14)
        self.embed_dim = self.clip.config.projection_dim
        
        # Classification head (parameters need a high LR)
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim * 2, 512),  # *2 because we concat image+text
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, 2)  # Binary classification
        )
        
        # Store parameter groups for the differential LR
        self.backbone_params = backbone_params
        self.classifier_params = self.classifier.parameters()
    
    def forward(self, pixel_values, input_ids, attention_mask):
        # Get CLIP embeddings
        outputs = self.clip(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get normalized embeddings
        image_embeds = outputs.image_embeds  # [batch, embed_dim]
        text_embeds = outputs.text_embeds    # [batch, embed_dim]
        
        # Concatenate image and text embeddings
        combined = torch.cat([image_embeds, text_embeds], dim=1)  # [batch, embed_dim*2]
        
        # Classify
        logits = self.classifier(combined)
        return logits

# Step 4: Create datasets and dataloaders
print("\nLoading CLIP processor and creating datasets...")
# CHANGED: Use processor for the larger model
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14") 

train_dataset = CLIPHateSpeechDataset(X_train, y_train, processor)
val_dataset = CLIPHateSpeechDataset(X_val, y_val, processor)
test_dataset = CLIPHateSpeechDataset(X_test, y_test, processor)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Step 5: Initialize model
print("\nInitializing CLIP-based classifier...")
# CHANGED: Pass unfreeze_layers=4 (e.g., fine-tune the last 4 blocks)
model = CLIPHateSpeechClassifier(unfreeze_layers=4) 
model = model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

# Step 6: Training setup (Major changes here for differential LR)
criterion = nn.CrossEntropyLoss()

# mess around with these numbers a bit
BACKBONE_LR = 1e-3
HEAD_LR = 1e-2

optimizer = torch.optim.AdamW([
    {'params': model.backbone_params, 'lr': BACKBONE_LR, 'weight_decay': 0.05},
    {'params': model.classifier_params, 'lr': HEAD_LR, 'weight_decay': 0.05}
])

num_epochs = 15 # play with this number a little bit
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs) 


# Training function (remains the same)
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(pixel_values, input_ids, attention_mask)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(loader), 100. * correct / total

# Evaluation function (remains the same)
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(pixel_values, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return total_loss / len(loader), accuracy * 100, f1, all_preds, all_labels

# Step 7: Training loop
print("\n" + "="*80)
print("Starting Training")
print("="*80)

best_val_acc = 0
train_losses = []
val_losses = []
train_accs = []
val_accs = []

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # Validate
    val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    # Learning rate schedule
    scheduler.step()
    
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f}")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        print(f"*** Saving best model with Val Acc: {best_val_acc:.2f}% to {MODEL_SAVE_PATH} ***")
        save_model(model, MODEL_SAVE_PATH) # Safetensors saving

print("\n" + "="*80)
print("Training Complete. Starting Final Evaluation.")
print("="*80)

## Step 8: Final Evaluation on Test Set 

# Load the best model weights using safetensors
if os.path.exists(MODEL_SAVE_PATH):
    # Use safetensors for loading (in-place)
    load_model(model, MODEL_SAVE_PATH) 
    print(f"Loaded best model from {MODEL_SAVE_PATH}")
    # CRITICAL: Ensure the model is on the correct device after loading
    model = model.to(device) 
else:
    print("Warning: Best model weights not found. Using final epoch weights for testing.")

# Evaluate on the test set
test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

print(f"\nFinal Test Loss: {test_loss:.4f}")
print(f"Final Test Accuracy: {test_acc:.2f}%")
print(f"Final Test F1-Score (Weighted): {test_f1:.4f}")

# Print Classification Report
target_names = ['Not Hate (0)', 'Hate Speech (1)']
print("\n--- Classification Report ---")
print(classification_report(test_labels, test_preds, target_names=target_names))


## Step 9: Visualization (No changes needed here)
# Plot training history
plt.figure(figsize=(12, 5))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Val Accuracy')
plt.title('Accuracy History')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('images/accuracyCLIP.png' ,dpi=300, bbox_inches='tight')

# Confusion Matrix Visualization
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    cbar=False,
    xticklabels=['Not Hate', 'Hate Speech'],
    yticklabels=['Not Hate', 'Hate Speech']
)
plt.title('Confusion Matrix on Test Set')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('images/confusionMatrixClip.png', dpi=300, bbox_inches='tight')

print("\nCode execution finished.")