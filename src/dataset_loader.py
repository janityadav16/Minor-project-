import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_flickr30k_dataset(base_path="data"):
    captions_file = os.path.join(base_path, "captions.txt")
    images_dir = os.path.join(base_path, "flickr30k_images")

    # --- Safety checks ---
    if not os.path.exists(captions_file):
        raise FileNotFoundError(f"❌ captions.txt not found at {captions_file}")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"❌ Image folder not found at {images_dir}")

    print("📂 Loading dataset...")

    # --- Load the CSV captions file ---
    df = pd.read_csv(captions_file)
    if not {'image_name', 'comment'}.issubset(df.columns):
        raise ValueError("❌ captions.txt must contain columns: image_name, comment")

    # --- Clean & prepare ---
    df = df.rename(columns={'image_name': 'image', 'comment': 'caption'})
    df['image'] = df['image'].apply(lambda x: str(x).strip())

    # --- Build full image path ---
    df['image_path'] = df['image'].apply(lambda x: os.path.join(images_dir, x))

    # --- Keep only valid image files ---
    df['exists'] = df['image_path'].apply(os.path.exists)
    valid_df = df[df['exists']].drop(columns='exists')

    print(f"✅ Found {len(valid_df)} valid image-caption pairs out of {len(df)} total")

    if len(valid_df) == 0:
        raise ValueError("❌ No valid image-caption pairs found. Check file names inside flickr30k_images/")

    # --- Split into train/test ---
    train_df, test_df = train_test_split(valid_df, test_size=0.2, random_state=42)

    # --- Save splits ---
    splits_dir = os.path.join(base_path, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    train_df.to_csv(os.path.join(splits_dir, "train_split.csv"), index=False)
    test_df.to_csv(os.path.join(splits_dir, "test_split.csv"), index=False)

    print(f"📊 Training: {len(train_df)} | Testing: {len(test_df)}")

    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = load_flickr30k_dataset()
    print("\n📈 Sample training data:")
    print(train_df.head())


import pickle
from collections import Counter
import nltk
nltk.download('punkt', quiet=True)

class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    def tokenizer(self, text):
        return nltk.tokenize.word_tokenize(text.lower())

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            tokens = self.tokenizer(sentence)
            frequencies.update(tokens)

        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer(text)
        return [
            self.stoi.get(token, self.stoi["<unk>"])
            for token in tokenized_text
        ]
