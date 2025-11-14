import json
import pandas as pd
import os
import re
import numpy as np
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch #

# transforms for the images
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5),          
])

def load_img(path, target_size):
    """Placeholder for loading image using PIL."""
    return Image.open(path).convert('RGB').resize(target_size)

def img_to_array(img):
    """Placeholder for converting PIL image to numpy array."""
    return np.array(img)


class MultimodalHateSpeechDataCleaner:
    """
    A class to clean and preprocess the Multimodal Hate Speech dataset.
    
    Attributes:
        base_path (str): Base path to the dataset
        df (pd.DataFrame): Main dataframe with all data
        X_train, X_val, X_test: Feature splits
        y_train, y_val, y_test: Label splits
    """
    
    def __init__(self, base_path, random_state=42):
        """
        Initialize the data cleaner.
        
        Args:
            base_path (str): Path to the dataset directory
            random_state (int): Random seed for reproducibility
        """
        self.base_path = base_path
        self.random_state = random_state
        self.df = None
        
        # Train/Val/Test splits
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        # Processed data
        self.X_train_text = None
        self.X_val_text = None
        self.X_test_text = None
        self.X_train_image = None
        self.X_val_image = None
        self.X_test_image = None
        
        # Label mappings
        self.label_mapping = {
            0: "NotHate",
            1: "Racist",
            2: "Sexist",
            3: "Homophobe",
            4: "Religion",
            5: "OtherHate"
        }
        
        self.hate_labels = ['Racist', 'Sexist', 'Homophobe', 'Religion', 'OtherHate']
        
    def load_annotations(self):
        """Load the JSON annotations file."""
        json_path = os.path.join(self.base_path, 'MMHS150K_GT.json')
        with open(json_path, 'r') as f:
            annotations = json.load(f)
        
        # Convert to DataFrame
        data = []
        for tweet_id, info in annotations.items():
            data.append({
                'tweet_id': tweet_id,
                'tweet_text': info['tweet_text'],
                'labels': info['labels'],
                'labels_str': info['labels_str']
            })
        
        self.df = pd.DataFrame(data)
        print(f"✅ Loaded {len(self.df)} annotations")
        return self
    
    def add_image_paths(self):
        """Add image path column to dataframe."""
        image_folder = os.path.join(self.base_path, 'img_resized')
        self.df['image_path'] = self.df['tweet_id'].apply(
            lambda x: os.path.join(image_folder, f"{x}.jpg")
        )
        print(f"✅ Added image paths")
        return self
    
    def create_labels(self):
        """Create majority labels and binary labels."""
        # Majority vote function
        def majority_vote(labels):
            label_count = Counter(labels)
            return label_count.most_common(1)[0][0]
        
        # Apply majority vote
        self.df['majority_label'] = self.df['labels'].apply(majority_vote)
        
        # Create string representation
        self.df['majority_label_str'] = self.df['majority_label'].map(self.label_mapping)
        
        # Create binary label (hate speech vs not)
        self.df['binary_label_str'] = self.df['majority_label_str'].apply(
            lambda x: 'hatespeech' if any(label in x for label in self.hate_labels) else 'not hatespeech'
        )
        
        self.df['binary_label'] = self.df['binary_label_str'].apply(
            lambda x: 1 if x == 'hatespeech' else 0
        )
        
        print(f"✅ Created labels")
        print(f"   Binary label distribution:\n{self.df['binary_label'].value_counts()}")
        print(f"   Majority label distribution:\n{self.df['majority_label'].value_counts()}")
        return self
    
    def balance_dataset(self, target_size=10000):
        """
        Balance the dataset based on the binary label (Hate vs NotHate), 
        aiming for a 1:1 ratio up to the specified target size.
        
        Args:
            target_size (int): The maximum number of samples to take for the 
                                minority (Hate Speech) class.
                                
        Note: The total size of the resulting balanced dataset will be 2 * target_size (if available).
        """
        
        # Calculate the total count of the Hate class (which is the minority)
        hate_count_total = self.df['binary_label'].value_counts()[1] # Label 1 is Hate
        
        # Determine the number of Hate samples to use: 
        # It's the smaller of the user-specified target_size or the total available count.
        sample_size_hate = min(hate_count_total, target_size)
        
        # We sample an equal number of Not Hate samples to maintain a 1:1 ratio
        sample_size_nothate = sample_size_hate 
        
        # --- Sampling ---
        df_hate_all = self.df[self.df['binary_label'] == 1]
        df_nothate_all = self.df[self.df['binary_label'] == 0]
        
        # Take the desired number of samples from each class
        df_hate = df_hate_all.sample(n=sample_size_hate, random_state=self.random_state)
        df_nothate = df_nothate_all.sample(n=sample_size_nothate, random_state=self.random_state)
        
        self.df = pd.concat([df_hate, df_nothate]).sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        print(f"✅ Balanced dataset for binary classification (Target Hate Samples: {target_size:,})")
        print(f"   Final total samples: {len(self.df):,}")
        print(f"   Final binary distribution:\n{self.df['binary_label'].value_counts().sort_index()}")
        return self

    def preprocess_text(self, text):
        """
        Preprocess text by lowercasing and removing URLs (minimal cleaning for LLMs).
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Cleaned text
        """
        text = text.lower()
        # Only remove URLs, keeping mentions, hashtags, punctuation, and numbers
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        return text
    
    def clean_text(self):
        """Apply text preprocessing to all tweets."""
        self.df['cleaned_text'] = self.df['tweet_text'].apply(self.preprocess_text)
        print(f"✅ Cleaned text data")
        return self
    
    def split_data(self, test_size=0.1, val_size=0.1):
        """
        Split data into train, validation, and test sets.
        
        Args:
            test_size (float): Proportion for test set
            val_size (float): Proportion of remaining data for validation
        """
        # Note: We still drop binary_label here, but the target label 
        # for training (majority_label_str) is in X.
        X = self.df.drop('binary_label', axis=1)
        y = self.df[['binary_label']] # y is still binary for the split metrics, but we use X's columns for Qwen format
        
        # Split into main + test
        X_main, self.X_test, y_main, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Split main into train + val
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_main, y_main, test_size=val_size, random_state=self.random_state
        )
        
        print(f"✅ Split data")
        print(f"   Training X size: {self.X_train.shape}")
        print(f"   Training y size: {self.y_train.shape}")
        print(f"   Validation X size: {self.X_val.shape}")
        print(f"   Validation y size: {self.y_val.shape}")
        print(f"   Testing X size: {self.X_test.shape}")
        print(f"   Testing y size: {self.y_test.shape}")

        # Extract the cleaned text column into the dedicated numpy array attributes
        self.X_train_text = self.X_train['cleaned_text'].values
        self.X_val_text = self.X_val['cleaned_text'].values
        self.X_test_text = self.X_test['cleaned_text'].values
        # -------------------------------------------
        
        return self

    # Logging Helper
    def _load_image_with_logging(self, img_path, target_size=(224, 224)):
        """
        Internal function to load an image, log status, and return array.
        
        Args:
            img_path (str): Path to image
            target_size (tuple): Target image size
            
        Returns:
            np.array: Preprocessed image array
        """
        try:
            if not os.path.exists(img_path):
                print(f"🚨 ERROR: Image file not found at: {img_path}")
                return np.zeros((target_size[0], target_size[1], 3))
            
            # Use original loading logic (assuming load_img/img_to_array are available)
            img = load_img(img_path, target_size=target_size)
            img_array = img_to_array(img) / 255.0
            
            # Logging success for 1 in 100 images to avoid spam
            if np.random.rand() < 0.01:
                 print(f"✅ DEBUG: Successfully loaded and processed image from: {os.path.basename(img_path)}")
            
            return img_array
        except Exception as e:
            print(f"❌ ERROR: Failed to load/preprocess image from: {img_path}. Error: {e}")
            return np.zeros((target_size[0], target_size[1], 3))

    # Modified Image Loading
    def load_and_preprocess_image(self, img_path, target_size=(224, 224)):
        """
        Load and preprocess a single image (now using the logging helper).
        """
        # Note: This method is kept for compatibility with the old method signature 
        # but now just calls the new logging wrapper.
        return self._load_image_with_logging(img_path, target_size)
    
    def load_images(self, target_size=(224, 224)):
        """
        Load and preprocess all images.
        
        Args:
            target_size (tuple): Target image size
        """
        print("Loading images... (this may take a while)")
        
        self.X_train_image = np.array([
            self.load_and_preprocess_image(path, target_size)
            for path in self.X_train['image_path']
        ])
        
        self.X_val_image = np.array([
            self.load_and_preprocess_image(path, target_size)
            for path in self.X_val['image_path']
        ])
        
        self.X_test_image = np.array([
            self.load_and_preprocess_image(path, target_size)
            for path in self.X_test['image_path']
        ])
        
        print(f"✅ Loaded images")
        print(f"   Train images shape: {self.X_train_image.shape}")
        print(f"   Val images shape: {self.X_val_image.shape}")
        print(f"   Test images shape: {self.X_test_image.shape}")
        return self
    
    def process_all(self, load_images=True, balance=True):
        """
        Run the complete data processing pipeline.
        
        Args:
            load_images (bool): Whether to load images (can be slow)
            balance (bool): Whether to balance the dataset
            
        Returns:
            self: The DataCleaner instance with all processed data
        """
        print("Starting data processing pipeline...")
        
        self.load_annotations()
        self.add_image_paths()
        self.create_labels()
        
        if balance:
            self.balance_dataset()
        
        self.clean_text()
        self.split_data()
        
        if load_images:
            self.load_images()
        
        print("\n✅ Data processing complete!")
        return self
    
    def get_data(self):
        """
        Get all processed data.
        
        Returns:
            dict: Dictionary containing all processed data
        """
        return {
            'X_train': self.X_train,
            'X_val': self.X_val,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_val': self.y_val,
            'y_test': self.y_test,
            'X_train_text': self.X_train_text,
            'X_val_text': self.X_val_text,
            'X_test_text': self.X_test_text,
            'X_train_image': self.X_train_image,
            'X_val_image': self.X_val_image,
            'X_test_image': self.X_test_image,
            'df': self.df
        }

# Helper to get specific split data for external Dataset creation
    def get_split_data(self, split='train'):
        """
        Get feature (X) and label (y) data for a specific split.
        
        Args:
            split (str): 'train', 'val', or 'test'
            
        Returns:
            tuple: (X_data, y_data)
        """
        if split == 'train':
            return self.X_train, self.y_train
        elif split == 'val':
            return self.X_val, self.y_val
        elif split == 'test':
            return self.X_test, self.y_test
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

    
    def get_qwen_format(self, split='train'):
        """
        Get data formatted for Qwen2-VL Dataset.
        
        Args:
            split (str): 'train', 'val', or 'test'
            
        Returns:
            dict: Dictionary with 'image', 'text', and 'label' keys
        """
        if split == 'train':
            X = self.X_train
            y = self.y_train
        elif split == 'val':
            X = self.X_val
            y = self.y_val
        elif split == 'test':
            X = self.X_test
            y = self.y_test
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")
        
        # Reset indices to ensure alignment
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        # Extract image paths
        image_paths = X['image_path'].tolist()
        
        # Format text as questions (binary question)
        texts = [
            f"Given the following text: '{text}'\n\nBased on the image and text above, is this hate speech (Yes or No)?"
            for text in X['cleaned_text']
        ]
        
        # Format labels as answers (binary answer)
        labels = [
            "Yes, this is hate speech." if label == 1 else "No, this is not hate speech."
            for label in y['binary_label']
        ]
        
        return {
            'image': image_paths,
            'text': texts,
            'label': labels
        }
    
    def get_all_qwen_format(self):
        """
        Get all splits formatted for Qwen2-VL.
        
        Returns:
            dict: Dictionary with 'train', 'val', and 'test' keys
        """
        return {
            'train': self.get_qwen_format('train'),
            'val': self.get_qwen_format('val'),
            'test': self.get_qwen_format('test')
        }
    
    def save_histograms(self, output_path):
        """
        Save histogram visualizations.
        
        Args:
            output_path (str): Path to save the histogram image
        """
        import matplotlib.pyplot as plt
        
        self.df.hist(figsize=(12, 10), bins=20)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved histogram to {output_path}")

    def save_training_example_image(self, output_dir, sample_index=0):
        """
        Saves a composite image of a training sample:
        1. The image after applying training augmentations.
        2. The associated cleaned text and labels.
        
        Args:
            output_dir (str): Directory to save the output image.
            sample_index (int): Index of the training sample to use (default is the first).
        """
        if self.X_train is None or self.y_train is None:
            print("🚨 ERROR: Training data not yet split. Run process_all() first.")
            return

        X_sample = self.X_train.iloc[sample_index]
        y_sample = self.y_train.iloc[sample_index]['binary_label']
        
        image_path = X_sample['image_path']
        cleaned_text = X_sample['cleaned_text']
        majority_label = X_sample['majority_label_str']
        binary_label = 'Hate Speech' if y_sample == 1 else 'Not Hate'
        
        # Load and augment image
        try:
            # Load as PIL Image for transformations
            img = Image.open(image_path).convert('RGB').resize((224, 224))
            # Convert to Tensor, apply transform, and convert back to PIL
            img_tensor = transforms.ToTensor()(img)
            augmented_tensor = train_transform(img_tensor)
            augmented_img = transforms.ToPILImage()(augmented_tensor)
            
        except Exception as e:
            print(f"❌ ERROR: Failed to load or augment image for example: {e}")
            return
        
        # Create composite image
        img_width, img_height = augmented_img.size
        # Set text canvas height based on content
        text_height = 200 
        composite_width = img_width
        composite_height = img_height + text_height
        
        composite_img = Image.new('RGB', (composite_width, composite_height), color='white')
        composite_img.paste(augmented_img, (0, 0))
        
        draw = ImageDraw.Draw(composite_img)
        
        # Try to use a default font path, otherwise fallback
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            print("Warning: Arial font not found. Using default PIL font.")
            font = ImageFont.load_default()
            
        text_y_start = img_height + 10
        
        # Formatting the text block
        text_block = (
            f"--- Training Sample {sample_index} ---\n"
            f"Binary Label: {binary_label}\n"
            f"Majority Label: {majority_label}\n"
            f"-----------------------------------\n"
            f"CLEANED TEXT:\n{cleaned_text}"
        )
        
        # Draw text (wrap text if necessary, though simpler is to just draw)
        draw.text((10, text_y_start), text_block, fill=(0, 0, 0), font=font)
        
        # Save the result
        output_path = os.path.join(output_dir, f"training_example_{sample_index}.png")
        os.makedirs(output_dir, exist_ok=True)
        composite_img.save(output_path)
        print(f"\n✨ Saved training sample image with augmentation and text to: {output_path}")


# Example usage
if __name__ == "__main__":
    BASE_PATH = '/projectnb/cs599m1/projects/multimodal-hatespeech/599_Project'
    OUTPUT_DIR = os.path.join(BASE_PATH, 'validation_examples')
    
    # Create cleaner instance and process all data
    cleaner = MultimodalHateSpeechDataCleaner(BASE_PATH, random_state=42)
    cleaner.process_all(load_images=True, balance=True)
    
    # Get all processed data
    data = cleaner.get_data()
    
    # Access individual components
    print(f"\nTrain text shape: {data['X_train_text'].shape}")
    print(f"Train image shape: {data['X_train_image'].shape}")
    print(f"Train labels shape: {data['y_train'].shape}")

    print("X test cleaned text example:", data['X_test']['cleaned_text'].iloc[0])
    print("X Val cleaned text example:", data['X_val']['cleaned_text'].iloc[0])

    # Save a sample of the processed training data
    cleaner.save_training_example_image(OUTPUT_DIR, sample_index=5) 

    # Save visualizations
    cleaner.save_histograms(os.path.join(BASE_PATH, 'images/histogram.png'))