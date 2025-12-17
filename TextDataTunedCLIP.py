"""
CLIP-based Hate Speech Classifier - TEXT ONLY
Uses only text embeddings from CLIP for classification
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
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
MODEL_SAVE_PATH = 'best_clip_text_only_classifier.safetensors' 

print("Processing data...")
cleaner = MultimodalHateSpeechDataCleaner(BASE_PATH, random_state=42)
cleaner.process_all(load_images=False, balance=True, target_size=20000) 

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

# Step 2: Create Dataset (TEXT ONLY)
class CLIPTextOnlyDataset(Dataset):
    def __init__(self, X, y, processor, max_length=77):
        self.texts = X['cleaned_text'].values
        self.labels = y['binary_label'].values
        self.processor = processor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Process text only
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Step 3: Define Model (TEXT ONLY)
class CLIPTextOnlyClassifier(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-large-patch14", unfreeze_layers=2):
        super().__init__()
        
        self.clip = CLIPModel.from_pretrained(
            clip_model_name,
            use_safetensors=True 
        )
        
        self.unfreeze_layers = unfreeze_layers
        backbone_params = []

        # Unfreeze last N layers of text encoder only
        text_layers = self.clip.text_model.encoder.layers
        for i, layer in enumerate(text_layers):
            if i >= len(text_layers) - self.unfreeze_layers:
                for param in layer.parameters():
                    param.requires_grad = True
                    backbone_params.append(param)
            else:
                for param in layer.parameters():
                    param.requires_grad = False
        
        # Freeze vision encoder completely
        for param in self.clip.vision_model.parameters():
            param.requires_grad = False
        
        self.embed_dim = self.clip.config.projection_dim
        
        # Classification head (only text embeddings)
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, 2)
        )
        
        self.backbone_params = backbone_params
        self.classifier_params = self.classifier.parameters()
    
    def forward(self, input_ids, attention_mask):
        # Get text embeddings only
        text_outputs = self.clip.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Classify
        logits = self.classifier(text_outputs)
        return logits

# Step 4: Create datasets and dataloaders
print("\nLoading CLIP processor and creating datasets...")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14") 

train_dataset = CLIPTextOnlyDataset(X_train, y_train, processor)
val_dataset = CLIPTextOnlyDataset(X_val, y_val, processor)
test_dataset = CLIPTextOnlyDataset(X_test, y_test, processor)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Step 5: Initialize model
print("\nInitializing CLIP-based TEXT ONLY classifier...")
model = CLIPTextOnlyClassifier(unfreeze_layers=4) 
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

# Step 6: Training setup
criterion = nn.CrossEntropyLoss()

BACKBONE_LR = 1e-3
HEAD_LR = 1e-2

optimizer = torch.optim.AdamW([
    {'params': model.backbone_params, 'lr': BACKBONE_LR, 'weight_decay': 0.05},
    {'params': model.classifier_params, 'lr': HEAD_LR, 'weight_decay': 0.05}
])

num_epochs = 15
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs) 

# Training function
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(loader), 100. * correct / total

# Evaluation function
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
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
print("Starting Training - TEXT ONLY MODEL")
print("="*80)

best_val_acc = 0
train_losses = []
val_losses = []
train_accs = []
val_accs = []

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    scheduler.step()
    
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        print(f"*** Saving best model with Val Acc: {best_val_acc:.2f}% to {MODEL_SAVE_PATH} ***")
        save_model(model, MODEL_SAVE_PATH)

print("\n" + "="*80)
print("Training Complete. Starting Final Evaluation.")
print("="*80)

# Step 8: Final Evaluation on Test Set
if os.path.exists(MODEL_SAVE_PATH):
    load_model(model, MODEL_SAVE_PATH) 
    print(f"Loaded best model from {MODEL_SAVE_PATH}")
    model = model.to(device) 
else:
    print("Warning: Best model weights not found. Using final epoch weights for testing.")

test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

print(f"\n=== TEXT ONLY MODEL RESULTS ===")
print(f"Final Test Loss: {test_loss:.4f}")
print(f"Final Test Accuracy: {test_acc:.2f}%")
print(f"Final Test F1-Score (Weighted): {test_f1:.4f}")

target_names = ['Not Hate (0)', 'Hate Speech (1)']
print("\n--- Classification Report ---")
print(classification_report(test_labels, test_preds, target_names=target_names))

# Step 9: Visualization
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Loss History - Text Only')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Val Accuracy')
plt.title('Accuracy History - Text Only')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('images/accuracyCLIP_text_only.png', dpi=300, bbox_inches='tight')

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
plt.title('Confusion Matrix - Text Only')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('images/confusionMatrixClip_text_only.png', dpi=300, bbox_inches='tight')

print("\nText-only model training finished.")