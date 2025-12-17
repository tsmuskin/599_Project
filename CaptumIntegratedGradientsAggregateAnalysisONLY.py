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
import pandas as pd

from DataCleanerClass import MultimodalHateSpeechDataCleaner

# ============================================================================
# SETUP
# ============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BASE_PATH = "/projectnb/cs599m1/projects/multimodal-hatespeech/599_Project"
OUTPUT_DIR = os.path.join(BASE_PATH, "captum_analysis_mahema")
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

print("Model loaded successfully!")

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

# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_all_samples_multimodal_only():
    """
    Run multimodal attribution analysis on ALL test samples.
    Only produces the aggregate_modality_analysis.png - no individual visualizations.
    """
    # Create new output directory
    AGGREGATE_OUTPUT_DIR = os.path.join(BASE_PATH, "AGGREGATEMULTIMODALCAPTUMANALYSIS")
    os.makedirs(AGGREGATE_OUTPUT_DIR, exist_ok=True)
    
    print(f"\nAnalyzing ALL {len(X_test)} test samples for multimodal model...\n")
    multimodal_results = {"image_contrib": [], "text_contrib": []}

    for idx in tqdm(range(len(X_test)), desc="Processing samples"):
        try:
            inputs, text, image, true_label = load_and_process_sample(idx, X_test, y_test, processor)

            # Get prediction
            pred_class, confidence = get_prediction(multimodal_model, inputs, "multimodal")

            # Compute attributions
            pixel_attr, text_attr = compute_multimodal_attributions(multimodal_model, inputs, pred_class)
            
            # Calculate contributions
            pixel_attr = pixel_attr.detach().squeeze(0)  # (3,H,W)
            text_attr = text_attr.detach().squeeze(0)    # (seq_len,embed_dim)

            pixel_importance = pixel_attr.abs().sum().item()
            text_importance_per_token = text_attr.abs().sum(dim=-1).cpu().numpy()
            text_importance_total = float(text_importance_per_token.sum())

            total = pixel_importance + text_importance_total + 1e-12
            image_contribution = (pixel_importance / total) * 100.0
            text_contribution = (text_importance_total / total) * 100.0

            multimodal_results["image_contrib"].append(image_contribution)
            multimodal_results["text_contrib"].append(text_contribution)

            # Print progress every 100 samples
            if (idx + 1) % 100 == 0:
                print(f"\nProcessed {idx + 1}/{len(X_test)} samples")
                print(f"  Current avg - Image: {np.mean(multimodal_results['image_contrib']):.1f}%, Text: {np.mean(multimodal_results['text_contrib']):.1f}%")

        except Exception as e:
            print(f"\n⚠️  Error processing sample {idx}: {e}")
            continue

    # ------------------- AGGREGATE MULTIMODAL RESULTS -------------------
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS - MULTIMODAL MODEL (ALL SAMPLES)")
    print("=" * 60)

    avg_image = float(np.mean(multimodal_results["image_contrib"])) if multimodal_results["image_contrib"] else 0.0
    avg_text = float(np.mean(multimodal_results["text_contrib"])) if multimodal_results["text_contrib"] else 0.0
    
    std_image = float(np.std(multimodal_results["image_contrib"])) if multimodal_results["image_contrib"] else 0.0
    std_text = float(np.std(multimodal_results["text_contrib"])) if multimodal_results["text_contrib"] else 0.0

    print(f"Total samples analyzed: {len(multimodal_results['image_contrib'])}")
    print(f"Average Image Contribution: {avg_image:.1f}% (±{std_image:.1f}%)")
    print(f"Average Text Contribution:  {avg_text:.1f}% (±{std_text:.1f}%)")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    axes[0].hist(multimodal_results["image_contrib"], bins=20, alpha=0.7, label="Image", color='blue')
    axes[0].hist(multimodal_results["text_contrib"], bins=20, alpha=0.7, label="Text", color='orange')
    axes[0].set_xlabel("Contribution (%)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"Distribution of Modality Contributions\n(n={len(multimodal_results['image_contrib'])} samples)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Bar chart
    axes[1].bar(["Image", "Text"], [avg_image, avg_text], color=['blue', 'orange'])
    axes[1].set_ylabel("Average Contribution (%)")
    axes[1].set_title("Average Modality Importance")
    axes[1].set_ylim([0, 100])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add error bars for standard deviation
    axes[1].errorbar(["Image", "Text"], [avg_image, avg_text], 
                     yerr=[std_image, std_text], 
                     fmt='none', ecolor='black', capsize=5, capthick=2)

    plt.tight_layout()
    output_path = os.path.join(AGGREGATE_OUTPUT_DIR, "aggregate_modality_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n✅ Aggregate visualization saved to: {output_path}")
    
    # Also save the raw data as CSV for further analysis
    results_df = pd.DataFrame(multimodal_results)
    csv_path = os.path.join(AGGREGATE_OUTPUT_DIR, "modality_contributions.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"✅ Raw data saved to: {csv_path}")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run analysis on ALL samples (multimodal only)
    analyze_all_samples_multimodal_only()
    print("\n✅ Multimodal attribution analysis complete!")