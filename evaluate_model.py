import os
import torch
from datasets import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from DataCleanerClass import MultimodalHateSpeechDataCleaner

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

BASE_PATH = "/projectnb/cs599m1/projects/multimodal-hatespeech/599_Project"
MODEL_DIR = "./qwen2vl-hate-speech-final"

# 1. Load processor and model
print("Loading processor and model...")
processor = Qwen2VLProcessor.from_pretrained(MODEL_DIR)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    dtype=torch.float16,
).eval().to("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load cleaned test data
print("Preparing test data...")
cleaner = MultimodalHateSpeechDataCleaner(BASE_PATH, random_state=42)
cleaner.process_all(balance=True)
qwen_data = cleaner.get_all_qwen_format()
test_dataset = Dataset.from_dict(qwen_data["test"])

# 3. Prepare input for model
def prepare_input(image_path, text):
    image = Image.open(image_path).convert("RGB")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text}
            ]
        }
    ]
    formatted_text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    return processor(text=[formatted_text], images=[image], return_tensors="pt", padding=True).to(model.device)

# 4. Run evaluation
print("Running evaluation...")
preds, labels = [], []

for i, example in enumerate(test_dataset):
    inputs = prepare_input(example["image"], example["text"])
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=10)
    pred_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    preds.append(pred_text.strip())
    labels.append(example["label"].strip())
    
    if i % 20 == 0:
        print(f"Processed {i}/{len(test_dataset)} samples")

# 5. Normalize labels
def normalize_label(text):
    t = text.lower()
    if "yes" in t:
        return "Yes, this is hate speech."
    elif "no" in t:
        return "No, this is not hate speech."
    return text.strip()

preds = [normalize_label(p) for p in preds]
labels = [normalize_label(l) for l in labels]

# 6. Compute metrics
accuracy = accuracy_score(labels, preds)
f1 = f1_score(labels, preds, average="weighted")
precision = precision_score(labels, preds, average="weighted")
recall = recall_score(labels, preds, average="weighted")

print("\n=== Evaluation Results ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

# 7. Confusion matrix
cm_labels = ["Yes, this is hate speech.", "No, this is not hate speech."]
cm = confusion_matrix(labels, preds, labels=cm_labels)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Yes", "No"], yticklabels=["Yes", "No"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Hate Speech Detection")
plt.tight_layout()
plt.savefig("images/confusion_matrix.png")
plt.show()

# 8. Save summary
with open("eval_results.txt", "w") as f:
    for p, l in zip(preds, labels):
        f.write(f"PRED: {p}\nTRUE: {l}\n\n")
    f.write("\n=== Evaluation Results ===\n")
    f.write(f"Accuracy:  {accuracy:.4f}\n")
    f.write(f"F1 Score:  {f1:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")