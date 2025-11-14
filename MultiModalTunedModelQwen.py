from datasets import Dataset
import os
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"
from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration, Trainer, TrainingArguments
from DataCleanerClass import MultimodalHateSpeechDataCleaner
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Step 1: Process data with cleaner
BASE_PATH = '/projectnb/cs599m1/projects/multimodal-hatespeech/599_Project'

print("Processing data...")
cleaner = MultimodalHateSpeechDataCleaner(BASE_PATH, random_state=42)

cleaner.process_all(load_images=False, balance=True) 

# Step 2: Get data in Qwen format
print("Converting to Qwen format...")
qwen_data = cleaner.get_all_qwen_format()

print(f"Train samples: {len(qwen_data['train']['image'])}")

print(f"Val samples: {len(qwen_data['val']['image'])}")

print(f"Test samples: {len(qwen_data['test']['image'])}")

# Example of what the data looks like:
print("\nExample train sample:")
print(f"Image path: {qwen_data['train']['image'][0]}")
print(f"Text: {qwen_data['train']['text'][0][:100]}...")
print(f"Label: {qwen_data['train']['label'][0]}") # Should now show a category (e.g., The category is NotHate.)

# Example of what the data looks like:
print("\nExample val sample label:")
print(f"{qwen_data['val']['label'][0]}")

print("\nExample test sample label ")
print(f"{qwen_data['test']['label'][0]}")

# Step 3: Create Hugging Face Datasets (NO PREPROCESSING)
train_dataset = Dataset.from_dict(qwen_data['train'])
val_dataset = Dataset.from_dict(qwen_data['val'])
test_dataset = Dataset.from_dict(qwen_data['test'])

# Step 4: Load processor
processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

#Step 5: Create the Collator: this does all of the tokenization of the examples, and masks the answer prompts so the model can train
class Qwen2VLCollator:
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, batch):
        images = []
        full_conversations = []
        
        for item in batch:
            try:
                img = Image.open(item['image']).convert('RGB')
                images.append(img)
                
                # Full conversation (Prompt + Answer)
                full_conv = [
                    {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": item['text']}]},
                    {"role": "assistant", "content": [{"type": "text", "text": item['label']}]}
                ]
                full_conversations.append(full_conv)
            except Exception as e:
                print(f"Error loading image {item['image']}: {e}")
                continue

        # 1. Apply Chat Template to get the formatted text strings
        # This is what Qwen2VLProcessor expects in the 'text' argument.
        formatted_texts = [
            self.processor.apply_chat_template(
                conv, 
                tokenize=False, 
                add_generation_prompt=False
            ) 
            for conv in full_conversations
        ]
        
        # 2. Tokenize and Batch the formatted texts and images
        model_inputs = self.processor(
            text=formatted_texts, # Pass the list of strings
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=False, # i turned this off bc this was causing a token count error
            max_length=2048 # Increased max_length to safely accommodate image tokens
        )
        
        # 3. Handle Labels for Instruction Tuning (Crucial for Qwen-VL)
        # We need to manually mask the prompt tokens with -100.
        labels = model_inputs["input_ids"].clone()
        
        # Re-tokenize the prompt-only strings to find the length for masking.
        prompt_conversations = [
            [conv[0]] for conv in full_conversations # Use only the user/prompt part
        ]


        # Convert the prompt conversations into text strings
        prompt_texts = [
            self.processor.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=False
            )
            for conv in prompt_conversations
        ]

        # Tokenize those prompt strings (this returns dict with attention_mask)
        prompt_inputs = self.processor(
            text=prompt_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=False,
            max_length=2048

        )
        
        
        # Mask the prompt tokens
        for i in range(labels.shape[0]):
            # Find the length of the prompt for the i-th sample (excluding padding)
            # Use attention mask to find the actual length of the prompt tokens
            prompt_len = (prompt_inputs['attention_mask'][i]).sum().item()

            # Mask the prompt tokens and the image placeholder tokens
            labels[i, :prompt_len] = -100
            
            # Optional: Ensure padding tokens are also masked
            padding_mask = (labels[i] == self.processor.tokenizer.pad_token_id)
            labels[i][padding_mask] = -100


        # model_inputs["labels"] = labels
        model_inputs["labels"] = labels.to(torch.long)

        return model_inputs

# Create collator
collator = Qwen2VLCollator(processor)

# DEBUG: Detailed inspection of one batch
print("\n" + "="*80)
print("🔍 DETAILED COLLATOR DEBUG")
print("="*80)

test_batch = [train_dataset[0]]  # Just one sample
test_output = collator(test_batch)

print(f"\n📊 Batch Statistics:")
print(f"Input IDs shape: {test_output['input_ids'].shape}")
print(f"Labels shape: {test_output['labels'].shape}")
print(f"Total tokens: {test_output['labels'].numel()}")
print(f"Non-masked tokens: {(test_output['labels'] != -100).sum().item()}")
print(f"Masked tokens: {(test_output['labels'] == -100).sum().item()}")
print(f"Percentage trainable: {(test_output['labels'] != -100).sum().item() / test_output['labels'].numel() * 100:.2f}%")

print(f"\n📝 Raw Input/Output:")
print(f"Question: {test_batch[0]['text'][:200]}...")
print(f"Expected answer: {test_batch[0]['label']}")

print(f"\n🔤 Tokenization Details:")
input_text = processor.tokenizer.decode(test_output['input_ids'][0])
print(f"Full tokenized text:\n{input_text}\n")

print(f"🎯 Label Tokens (what model trains on):")
label_tokens = test_output['labels'][0]
trainable_indices = (label_tokens != -100).nonzero(as_tuple=True)[0]
if len(trainable_indices) > 0:
    trainable_tokens = test_output['input_ids'][0][trainable_indices]
    decoded_labels = processor.tokenizer.decode(trainable_tokens)
    print(f"Decoded trainable portion: {decoded_labels}")
    print(f"Number of trainable tokens: {len(trainable_indices)}")
else:
    print("⚠️  ERROR: NO TRAINABLE TOKENS! All labels are masked!")

print("="*80 + "\n")

# clear the cuda cache in case this helps
torch.cuda.empty_cache()

# Configure Lora - try to optimize fine tuning
print('configure lora')
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)


# # Step 6: Load model
print("Loading Qwen2-VL model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.float16,
)

# # Apply lora to model
model.config.use_cache = False
model.enable_input_require_grads()
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()



# Step 7: Setup training
training_args = TrainingArguments(
    output_dir="./qwen2vl-hate-speech-mahema",
    gradient_checkpointing=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    fp16=True,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    # Add this if using device_map="auto"
    # device="cuda:0",  # Specify device explicitly
)

# Step 8: Create trainer with custom collator
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,      # Raw dataset - no preprocessing!
    eval_dataset=val_dataset,          # Raw dataset - no preprocessing!
    data_collator=collator,            # Collator does all the work
)

# clear the cache one more time?
torch.cuda.empty_cache()
# Step 9: Train
print("Starting training...")
# trainer.train()
trainer.train()

# Step 10: Evaluate
print("Evaluating on val set...")
val_results = trainer.evaluate(val_dataset)
print(f"Val results: {val_results}")

print("Evaluating on test set...")
test_results = trainer.evaluate(test_dataset)
print(f"Test results: {test_results}")


# Step 11: Save model

model = model.merge_and_unload() 
model.save_pretrained("./qwen2vl-hate-speech-final")
processor.tokenizer.save_pretrained("./qwen2vl-hate-speech-final")
processor.save_pretrained("./qwen2vl-hate-speech-final")
trainer.save_model("./qwen2vl-hate-speech-final")