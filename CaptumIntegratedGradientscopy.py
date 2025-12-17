import os
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import CLIPProcessor, CLIPModel
from captum.attr import IntegratedGradients, LayerIntegratedGradients
from safetensors.torch import load_model
import matplotlib.pyplot as plt

from DataCleanerClass import MultimodalHateSpeechDataCleaner
import matplotlib as mpl
mpl.rcParams["text.parse_math"] = False


# ============================================================================
# SETUP
# ============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BASE_PATH = "/projectnb/cs599m1/projects/multimodal-hatespeech/599_Project"
OUTPUT_DIR = os.path.join(BASE_PATH, "captum_analysis_all")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class CLIPHateSpeechClassifier(nn.Module):
    """
    Multimodal classifier: image + text
    Uses CLIP pooled image & text features (get_*_features).
    """
    def __init__(self, clip_model_name="openai/clip-vit-large-patch14"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name, use_safetensors=True)
        self.embed_dim = self.clip.config.projection_dim

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, 2),
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        """
        Forward pass:
        - image_embeds: (B, D)
        - text_embeds:  (B, D)
        - concat -> (B, 2D) -> logits (B, 2)
        """
        image_embeds = self.clip.get_image_features(pixel_values=pixel_values)  # (B, D)
        text_embeds = self.clip.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )  # (B, D)

        combined = torch.cat([image_embeds, text_embeds], dim=1)  # (B, 2D)
        logits = self.classifier(combined)  # (B, 2)
        return logits


class CLIPTextOnlyClassifier(nn.Module):
    """Text-only classifier on top of CLIP text features."""
    def __init__(self, clip_model_name="openai/clip-vit-large-patch14"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name, use_safetensors=True)
        self.embed_dim = self.clip.config.projection_dim

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, 2),
        )

    def forward(self, input_ids, attention_mask):
        text_features = self.clip.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )  # (B, D)
        logits = self.classifier(text_features)  # (B, 2)
        return logits


class CLIPImageOnlyClassifier(nn.Module):
    """Image-only classifier on top of CLIP image features."""
    def __init__(self, clip_model_name="openai/clip-vit-large-patch14"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name, use_safetensors=True)
        self.embed_dim = self.clip.config.projection_dim

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, 2),
        )

    def forward(self, pixel_values):
        image_features = self.clip.get_image_features(pixel_values=pixel_values)  # (B, D)
        logits = self.classifier(image_features)  # (B, 2)
        return logits

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading test data...")
cleaner = MultimodalHateSpeechDataCleaner(BASE_PATH, random_state=42)
cleaner.process_all(load_images=False, balance=True)
data = cleaner.get_data()

X_test = data["X_test"]
y_test = data["y_test"]

processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

print("\nLoading trained models...")

multimodal_model = CLIPHateSpeechClassifier()
load_model(multimodal_model, "best_clip_classifier_2.safetensors")
multimodal_model = multimodal_model.to(device)
multimodal_model.eval()

text_model = CLIPTextOnlyClassifier()
load_model(text_model, "best_clip_text_only_classifier.safetensors")
text_model = text_model.to(device)
text_model.eval()

image_model = CLIPImageOnlyClassifier()
load_model(image_model, "best_clip_image_only_classifier.safetensors")
image_model = image_model.to(device)
image_model.eval()

print("All models loaded successfully!")

# ============================================================================
# HELPERS: DATA LOADING & PREDICTION
# ============================================================================

def load_and_process_sample(idx, X_test, y_test, processor):
    """Load a single sample (text + image) and preprocess with CLIPProcessor."""
    image_path = X_test.iloc[idx]["image_path"]
    text = X_test.iloc[idx]["cleaned_text"]
    label = y_test.iloc[idx]["binary_label"]

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        image = Image.new("RGB", (224, 224), color="white")

    inputs = processor(
        text=[text],
        images=image,
        return_tensors="pt",
        padding="max_length",
        max_length=77,
        truncation=True,
    )

    return inputs, text, image, label


def get_prediction(model, inputs, model_type="multimodal"):
    """Return predicted class + confidence for a given model and CLIP-processed inputs."""
    with torch.no_grad():
        if model_type == "multimodal":
            outputs = model(
                inputs["pixel_values"].to(device),
                inputs["input_ids"].to(device),
                inputs["attention_mask"].to(device),
            )
        elif model_type == "text":
            outputs = model(
                inputs["input_ids"].to(device),
                inputs["attention_mask"].to(device),
            )
        elif model_type == "image":
            outputs = model(inputs["pixel_values"].to(device))
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    probs = torch.softmax(outputs, dim=1)  # (1,2)
    pred_class = torch.argmax(probs, dim=1).item()
    return pred_class, probs[0, pred_class].item()

# ============================================================================
# ATTRIBUTION FUNCTIONS
# ============================================================================

def compute_multimodal_attributions(model, inputs, target_class):
    """
    Multimodal attributions:
      - Image: IntegratedGradients on pixel_values
      - Text:  LayerIntegratedGradients on the token embedding layer
    Returns:
      pixel_attr: (1, 3, H, W)
      text_attr:  (1, seq_len, embed_dim)
    """
    model.eval()

    pixel_values = inputs["pixel_values"].to(device)      # (1,3,H,W)
    input_ids = inputs["input_ids"].to(device)            # (1,seq_len)
    attention_mask = inputs["attention_mask"].to(device)  # (1,seq_len)

    # ----------------------------------------------------
    # 1. IMAGE IG
    # ----------------------------------------------------
    pixel_values_ig = pixel_values.clone().detach().requires_grad_(True)
    baseline_pixel = torch.zeros_like(pixel_values_ig)

    def forward_pixel(px):
        B_ig = px.size(0)
        ids_exp = input_ids.expand(B_ig, -1)
        mask_exp = attention_mask.expand(B_ig, -1)
        logits = model(px, ids_exp, mask_exp)  # (B_ig, 2)
        return logits

    ig = IntegratedGradients(forward_pixel)
    pixel_attr = ig.attribute(
        inputs=pixel_values_ig,
        baselines=baseline_pixel,
        target=target_class,
        return_convergence_delta=False,
    )  # (1,3,H,W)

    # ----------------------------------------------------
    # 2. TEXT LAYER IG (token embeddings)
    # ----------------------------------------------------
    text_model = model.clip.text_model

    # ★★ CLip usees this layer★★
    if hasattr(text_model, "embeddings") and hasattr(text_model.embeddings, "token_embedding"):
        embedding_layer = text_model.embeddings.token_embedding
    else:
        raise RuntimeError(
            "Cannot locate CLIP token embedding layer. "
            "Expected clip.text_model.embeddings.token_embedding."
        )

    def forward_text(input_ids_LIG):
        B_lig = input_ids_LIG.size(0)

        # Text features from CLIP
        mask_exp = attention_mask.expand(B_lig, -1)
        text_features = model.clip.get_text_features(
            input_ids=input_ids_LIG,
            attention_mask=mask_exp,
        )  # (B_lig, D)

        # Image features (same image for all B_lig samples)
        pixel_exp = pixel_values.expand(B_lig, -1, -1, -1)
        image_features = model.clip.get_image_features(pixel_values=pixel_exp)  # (B_lig, D)

        combined = torch.cat([image_features, text_features], dim=1)  # (B_lig, 2D)
        logits = model.classifier(combined)  # (B_lig, 2)
        return logits

    lig = LayerIntegratedGradients(forward_text, embedding_layer)

    baseline_ids = torch.zeros_like(input_ids)
    text_attr = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        target=target_class,
        return_convergence_delta=False,
    )  # (1, seq_len, embed_dim)

    return pixel_attr, text_attr


def compute_text_attributions(model, inputs, target_class):
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    text_model = model.clip.text_model

    # -------------------------
    # 1. Correct embedding layer (OpenAI CLIP compatible)
    # -------------------------
    if hasattr(text_model, "embeddings") and hasattr(text_model.embeddings, "token_embedding"):
        embedding_layer = text_model.embeddings.token_embedding
    else:
        raise RuntimeError("Cannot find token embedding layer in CLIP text model.")

    # -------------------------
    # 2. Forward function for Layer IG
    # -------------------------
    def forward_text(input_ids_):
        # OpenAI CLIP only accepts input_ids
        outputs = text_model(input_ids_)

        # last_hidden = outputs[0]
        last_hidden = outputs[0]  # (B, seq_len, dim)

        eos_id = model.clip.config.text_config.eos_token_id
        eos_mask = (input_ids_ == eos_id)

        if not eos_mask.any():
            eos_positions = torch.full((input_ids_.size(0),), input_ids_.size(1)-1, device=input_ids_.device)
        else:
            eos_positions = eos_mask.float().argmax(dim=1)

        batch_idx = torch.arange(input_ids_.size(0), device=input_ids_.device)

        pooled = last_hidden[batch_idx, eos_positions]  # (B, dim)

        logits = model.classifier(pooled)
        return logits[:, target_class]

    # -------------------------
    # 3. Build Layer IG
    # -------------------------
    lig = LayerIntegratedGradients(forward_text, embedding_layer)

    # -------------------------
    # 4. Baseline = PAD token ID
    # -------------------------
    pad_id = model.clip.config.text_config.pad_token_id
    if pad_id is None:
        pad_id = 0
    baseline_ids = torch.full_like(input_ids, pad_id)

    # -------------------------
    # 5. Compute attributions
    # -------------------------
    text_attr = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        return_convergence_delta=False,
    )

    return text_attr

def compute_image_attributions(model, inputs, target_class):
    pixel_values = inputs["pixel_values"].to(device)

    # Enable gradient
    pixel_values_ig = pixel_values.clone().detach().requires_grad_(True)

    def forward_img(px):
        # full CLIP vision transformer with gradients
        vision_outputs = model.clip.vision_model(
            pixel_values=px,
            output_hidden_states=True
        )
        
        # final hidden states (B, seq_len, 1024)
        last_hidden = vision_outputs.hidden_states[-1]

        # CLS token (position 0)
        pooled = last_hidden[:, 0, :]  # (B, 1024)

        # Project to 768-dim (same as get_image_features output)
        projected = model.clip.visual_projection(pooled)  # (B, 768)

        logits = model.classifier(projected)
        return logits[:, target_class]

    ig = IntegratedGradients(forward_img)

    baseline = torch.zeros_like(pixel_values_ig)

    pixel_attr = ig.attribute(
        inputs=pixel_values_ig,
        baselines=baseline,
        return_convergence_delta=False,
    )
    return pixel_attr



# ============================================================================
# VISUALISATION FUNCTIONS
# ============================================================================

def visualize_multimodal_comparison(
    pixel_attr,
    text_attr,
    text,
    tokens,
    image,
    sample_idx,
    output_dir,
    k=100
):
    """
    Visualize:
      - modality contributions (image vs text)
      - top-20 token attributions
      - bounding box around top-K pixels (CHANGED from heatmap)
      - original image
    """
    import matplotlib.patches as patches
    
    pixel_attr = pixel_attr.detach().squeeze(0)  # (3,H,W)
    text_attr = text_attr.detach().squeeze(0)    # (seq_len,embed_dim)

    # Image importance (sum over channels + spatial)
    pixel_importance = pixel_attr.abs().sum().item()

    # Per-token importance (sum over embedding dim)
    text_importance_per_token = text_attr.abs().sum(dim=-1).cpu().numpy()
    text_importance_total = float(text_importance_per_token.sum())

    total = pixel_importance + text_importance_total + 1e-12
    image_contribution = (pixel_importance / total) * 100.0
    text_contribution = (text_importance_total / total) * 100.0

    # Filter tokens (remove special)
    valid_tokens = []
    valid_scores = []
    for tok, score in zip(tokens, text_importance_per_token):
        if tok not in ["<|startoftext|>", "<|endoftext|>", "[PAD]"]:
            valid_tokens.append(tok)
            valid_scores.append(float(score))

    pairs = sorted(zip(valid_tokens, valid_scores), key=lambda x: x[1], reverse=True)
    if len(pairs) > 0:
        top_tokens, top_scores = zip(*pairs[:20])
    else:
        top_tokens, top_scores = [], []

    # === NEW: Calculate bounding box ===
    pixel_importance_map = pixel_attr.abs().sum(dim=0).cpu().numpy()  # (H,W)
    flat = pixel_importance_map.flatten()
    top_k_indices = np.argsort(flat)[-k:]
    top_k_coords = np.unravel_index(top_k_indices, pixel_importance_map.shape)
    
    # Get bounding box coordinates
    rows, cols = top_k_coords
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()
    
    # Add padding to bounding box (5% of image size)
    h, w = pixel_importance_map.shape
    padding_h = int(h * 0.05)
    padding_w = int(w * 0.05)
    
    min_row = max(0, min_row - padding_h)
    max_row = min(h - 1, max_row + padding_h)
    min_col = max(0, min_col - padding_w)
    max_col = min(w - 1, max_col + padding_w)
    
    bbox_height = max_row - min_row
    bbox_width = max_col - min_col

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Modality contributions
    axes[0, 0].bar(["Image", "Text"], [image_contribution, text_contribution])
    axes[0, 0].set_ylabel("Attribution (%)")
    axes[0, 0].set_ylim([0, 100])
    axes[0, 0].set_title("Modality Importance Comparison")

    # Token-level attributions
    axes[0, 1].barh(range(len(top_tokens)), top_scores)
    axes[0, 1].set_yticks(range(len(top_tokens)))
    axes[0, 1].set_yticklabels(top_tokens, fontsize=8)
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel("Attribution Score")
    axes[0, 1].set_title("Top 20 Token Attributions")

    # === CHANGED: Bounding box instead of heatmap ===
    axes[1, 0].imshow(image)
    rect = patches.Rectangle(
        (min_col, min_row), 
        bbox_width, 
        bbox_height,
        linewidth=3, 
        edgecolor='lime', 
        facecolor='none',
        linestyle='-'
    )
    axes[1, 0].add_patch(rect)
    axes[1, 0].set_title(f"Bounding Box Around Top {k} Pixels")
    axes[1, 0].axis("off")

    # Original image
    axes[1, 1].imshow(image)
    axes[1, 1].set_title("Original Image")
    axes[1, 1].axis("off")

    plt.suptitle(f"Multimodal IG — Sample {sample_idx}\nText: \"{text[:80]}...\"", fontsize=10)
    plt.tight_layout()
    fname = os.path.join(output_dir, f"multimodal_sample_{sample_idx}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()

    return image_contribution, text_contribution

def visualize_top_k_tokens(
    model,
    text_attr,
    inputs,
    text,
    tokens,
    k=10,
    sample_idx=0,
    output_dir=""
):
    """
    FIXED VERSION:
    - filters only *real* BPE tokens
    - strips </w> artifacts
    - avoids PAD/SOT/EOT tokens
    - sorts by attribution magnitude
    """

    text_attr = text_attr.detach().squeeze(0)   # (seq_len, embed_dim)

    # compute per-token scores
    per_token_scores = text_attr.abs().sum(dim=-1).cpu().numpy()

    cleaned = []
    for tok, score in zip(tokens, per_token_scores):

        # Skip special tokens
        if tok in ["<|startoftext|>", "<|endoftext|>", "[PAD]"]:
            continue
        
        # Strip </w>
        tok_clean = tok.replace("</w>", "")

        # Skip garbage empty tokens
        if tok_clean.strip() == "":
            continue

        cleaned.append((tok_clean, float(score)))

    # sort tokens by importance
    cleaned.sort(key=lambda x: x[1], reverse=True)

    # take top k
    top_k = cleaned[:k] if cleaned else []

    if not top_k:
        print(f"No valid tokens found for sample {sample_idx}")
        return []

    plot_tokens = [t for t, _ in top_k]
    plot_scores = [s for _, s in top_k]

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(range(len(plot_tokens)), plot_scores)
    ax.set_yticks(range(len(plot_tokens)))
    ax.set_yticklabels(plot_tokens)
    ax.invert_yaxis()
    ax.set_xlabel("Attribution Score")
    ax.set_title(f"Top {k} Token Attributions — Sample {sample_idx}")

    plt.suptitle(f"Multimodal IG — Sample {sample_idx}\nText: \"{text}...\"", fontsize=10)
    plt.tight_layout()
    fname = os.path.join(output_dir, f"top_{k}_tokens_sample_{sample_idx}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()

    return top_k

def visualize_top_k_pixels_with_bbox(pixel_attr, image, k=100, sample_idx=0, output_dir=""):
    """
    Visualize top-k most important pixels with a bounding box around the region.
    Creates a 2x2 grid showing:
    1. Original image
    2. Attribution heatmap
    3. Bounding box around top-K region
    4. Combined: bounding box + heatmap overlay
    """
    import matplotlib.patches as patches
    
    pixel_attr = pixel_attr.detach().squeeze(0)  # (3,H,W)
    pixel_importance = pixel_attr.abs().sum(dim=0).cpu().numpy()  # (H,W)

    # Find top K pixels
    flat = pixel_importance.flatten()
    top_k_indices = np.argsort(flat)[-k:]
    top_k_coords = np.unravel_index(top_k_indices, pixel_importance.shape)
    
    # Get bounding box coordinates
    rows, cols = top_k_coords
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()
    
    # Add padding to bounding box (5% of image size)
    h, w = pixel_importance.shape
    padding_h = int(h * 0.05)
    padding_w = int(w * 0.05)
    
    min_row = max(0, min_row - padding_h)
    max_row = min(h - 1, max_row + padding_h)
    min_col = max(0, min_col - padding_w)
    max_col = min(w - 1, max_col + padding_w)
    
    bbox_height = max_row - min_row
    bbox_width = max_col - min_col
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # 1. Original Image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[0, 0].axis("off")
    
    # 2. Attribution Heatmap
    axes[0, 1].imshow(pixel_importance, cmap="hot")
    axes[0, 1].set_title("Attribution Heatmap", fontsize=12, fontweight='bold')
    axes[0, 1].axis("off")
    
    # 3. Bounding Box on Original Image
    axes[1, 0].imshow(image)
    rect = patches.Rectangle(
        (min_col, min_row), 
        bbox_width, 
        bbox_height,
        linewidth=3, 
        edgecolor='lime', 
        facecolor='none',
        linestyle='-'
    )
    axes[1, 0].add_patch(rect)
    axes[1, 0].set_title(f"Bounding Box Around Top {k} Pixels", fontsize=12, fontweight='bold')
    axes[1, 0].axis("off")
    
    # 4. Combined: Bounding Box + Heatmap Overlay
    axes[1, 1].imshow(image)
    axes[1, 1].imshow(pixel_importance, cmap="hot", alpha=0.4)
    rect2 = patches.Rectangle(
        (min_col, min_row), 
        bbox_width, 
        bbox_height,
        linewidth=3, 
        edgecolor='lime', 
        facecolor='none',
        linestyle='-'
    )
    axes[1, 1].add_patch(rect2)
    axes[1, 1].set_title("Combined: BBox + Heatmap", fontsize=12, fontweight='bold')
    axes[1, 1].axis("off")
    
    plt.suptitle(f"Top-K Pixel Attribution with Bounding Box — Sample {sample_idx}", 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    fname = os.path.join(output_dir, f"bbox_top_{k}_pixels_sample_{sample_idx}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    
    # Print bbox info
    print(f"  Bounding box: [{min_row}:{max_row}, {min_col}:{max_col}]")
    print(f"  Box dimensions: {bbox_height}x{bbox_width} pixels")

def visualize_top_k_pixels(pixel_attr, image, k=100, sample_idx=0, output_dir=""):
    """
    Visualize top-k most important pixels (pixel_attr: (1,3,H,W)).
    """
    pixel_attr = pixel_attr.detach().squeeze(0)  # (3,H,W)
    pixel_importance = pixel_attr.abs().sum(dim=0).cpu().numpy()  # (H,W)

    flat = pixel_importance.flatten()
    top_k_indices = np.argsort(flat)[-k:]
    mask = np.zeros_like(pixel_importance)
    top_k_coords = np.unravel_index(top_k_indices, pixel_importance.shape)
    mask[top_k_coords] = 1

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(pixel_importance, cmap="hot")
    axes[1].set_title("Attribution Heatmap")
    axes[1].axis("off")

    axes[2].imshow(image)
    axes[2].imshow(mask, alpha=0.5, cmap="hot")
    axes[2].set_title(f"Top {k} Important Pixels")
    axes[2].axis("off")

    plt.suptitle(f"Image Attribution Analysis — Sample {sample_idx}")
    plt.tight_layout()
    fname = os.path.join(output_dir, f"top_{k}_pixels_sample_{sample_idx}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()

# ============================================================================
# MAIN ANALYSIS LOOP
# ============================================================================

def analyze_samples(num_samples=len(X_test), k_tokens=10, k_pixels=100):
    print(f"\nAnalyzing {num_samples} samples...\n")
    multimodal_results = {"image_contrib": [], "text_contrib": []}

    for idx in tqdm(range(len(X_test))):
        print("\n" + "=" * 60)
        print(f"Analyzing sample {idx}")
        print("=" * 60)

        inputs, text, image, true_label = load_and_process_sample(idx, X_test, y_test, processor)

        tokens = processor.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu().numpy())
        print(f"Text: {text}")
        print(f"True Label: {'Hate Speech' if true_label == 1 else 'Not Hate'}")

        # ------------------- MULTIMODAL -------------------
        print("\n--- Multimodal Model ---")
        pred_class, confidence = get_prediction(multimodal_model, inputs, "multimodal")
        print(f"Prediction: {'Hate Speech' if pred_class == 1 else 'Not Hate'} "
              f"(confidence: {confidence:.2%})")

        pixel_attr, text_attr = compute_multimodal_attributions(multimodal_model, inputs, pred_class)
        image_contrib, text_contrib = visualize_multimodal_comparison(pixel_attr, text_attr, text, tokens, image, idx, OUTPUT_DIR, k=k_pixels)

        multimodal_results["image_contrib"].append(image_contrib)
        multimodal_results["text_contrib"].append(text_contrib)

        print(f"Image contribution: {image_contrib:.1f}%")
        print(f"Text contribution:  {text_contrib:.1f}%")

        # ------------------- TEXT-ONLY -------------------
        print("\n--- Text-Only Model ---")
        pred_class_text, confidence_text = get_prediction(text_model, inputs, "text")
        print(f"Prediction: {'Hate Speech' if pred_class_text == 1 else 'Not Hate'} "
              f"(confidence: {confidence_text:.2%})")

        text_attr_only = compute_text_attributions(text_model, inputs, pred_class_text)
        top_tokens = visualize_top_k_tokens(model=text_model, text_attr=text_attr_only, inputs=inputs,
                    text=text, tokens=tokens, k=k_tokens, sample_idx=idx, output_dir=OUTPUT_DIR)

        print(f"Top 5 tokens: {[t[0] for t in top_tokens[:5]]}")

        # ------------------- IMAGE-ONLY -------------------
        print("\n--- Image-Only Model ---")
        pred_class_image, confidence_image = get_prediction(image_model, inputs, "image")
        print(f"Prediction: {'Hate Speech' if pred_class_image == 1 else 'Not Hate'} "
              f"(confidence: {confidence_image:.2%})")

        pixel_attr_image = compute_image_attributions(image_model, inputs, pred_class_image)
        visualize_top_k_pixels(pixel_attr_image, image, k=k_pixels, sample_idx=idx, output_dir=OUTPUT_DIR)
        visualize_top_k_pixels_with_bbox(pixel_attr_image, image, k=k_pixels, sample_idx=idx, output_dir=OUTPUT_DIR)

    # ------------------- AGGREGATE MULTIMODAL RESULTS -------------------
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS - MULTIMODAL MODEL")
    print("=" * 60)

    avg_image = float(np.mean(multimodal_results["image_contrib"])) if multimodal_results["image_contrib"] else 0.0
    avg_text = float(np.mean(multimodal_results["text_contrib"])) if multimodal_results["text_contrib"] else 0.0

    print(f"Average Image Contribution: {avg_image:.1f}%")
    print(f"Average Text Contribution:  {avg_text:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(multimodal_results["image_contrib"], bins=10, alpha=0.7, label="Image")
    axes[0].hist(multimodal_results["text_contrib"], bins=10, alpha=0.7, label="Text")
    axes[0].set_xlabel("Contribution (%)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Modality Contributions")
    axes[0].legend()

    axes[1].bar(["Image", "Text"], [avg_image, avg_text])
    axes[1].set_ylabel("Average Contribution (%)")
    axes[1].set_title("Average Modality Importance")
    axes[1].set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "aggregate_modality_analysis_all.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nAll visualizations saved to: {OUTPUT_DIR}")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    analyze_samples(num_samples=len(X_test), k_tokens=10, k_pixels=100)
    print("\n✅ Attribution analysis complete!")
