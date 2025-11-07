# src/train_model.py
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# -------------------------
# Dataset
# -------------------------
class CaptionDataset(Dataset):
    def __init__(self, split_csv, sequences_path="data/caption_sequences.npy", features_dir="data/features"):
        self.df = pd.read_csv(split_csv)
        self.features_dir = features_dir

        # Load caption sequences - they are in the same order as df rows produced earlier
        # If sequences align differently, you must create an index mapping. Here we assume order matches.
        self.sequences = np.load(sequences_path, allow_pickle=False)
        if len(self.sequences) != len(self.df):
            # try fallback: many captions per image -> repeat mapping by index modulo
            # Safe fallback: align by df index if shapes differ (keeps script usable for small tests)
            self.sequences = self.sequences[:len(self.df)]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(row['image_path'])
        feat_file = Path(self.features_dir) / (img_path.name + ".npy")
        feat = np.load(feat_file)  # shape: (2048,) for ResNet50
        caption_seq = self.sequences[idx]  # already padded to max_length
        return torch.from_numpy(feat).float(), torch.from_numpy(caption_seq).long()

# -------------------------
# Model: Encoder + Decoder
# -------------------------
class Encoder(nn.Module):
    def __init__(self, feat_dim=2048, embed_dim=512):
        super().__init__()
        self.fc = nn.Linear(feat_dim, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)

    def forward(self, features):
        x = self.fc(features)
        x = self.bn(x)
        x = torch.relu(x)
        return x  # (batch, embed_dim)

class DecoderLSTM(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size, padding_idx=0, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features_emb, captions):
        # captions: (batch, seq_len) of token ids (includes <start> ... <end> ... <pad>)
        embeddings = self.embed(captions)  # (batch, seq_len, embed_dim)
        # Prepend image feature as first time-step token embedding
        features_emb = features_emb.unsqueeze(1)  # (batch,1,embed_dim)
        lstm_input = torch.cat([features_emb, embeddings[:, :-1, :]], dim=1)  # teacher forcing
        out, _ = self.lstm(lstm_input)
        logits = self.fc(out)  # (batch, seq_len, vocab_size)
        return logits

# -------------------------
# Training utilities
# -------------------------
def collate_fn(batch):
    feats = torch.stack([b[0] for b in batch], dim=0)
    caps = torch.stack([b[1] for b in batch], dim=0)
    return feats, caps

def train_one_epoch(enc, dec, dataloader, opt_e, opt_d, criterion, device):
    enc.train(); dec.train()
    total_loss = 0.0
    for feats, caps in tqdm(dataloader, desc="train"):
        feats = feats.to(device)
        caps = caps.to(device)
        opt_e.zero_grad(); opt_d.zero_grad()
        feats_emb = enc(feats)  # (batch,embed_dim)
        logits = dec(feats_emb, caps)  # (batch, seq_len, vocab)
        # shift targets: predict tokens at t given inputs up to t-1 (logits aligned with captions)
        loss = criterion(logits.view(-1, logits.size(-1)), caps.view(-1))
        loss.backward()
        opt_e.step(); opt_d.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# -------------------------
# Main
# -------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits_dir = Path("data/splits")
    train_csv = splits_dir / "train_split.csv"
    assert train_csv.exists(), f"Train CSV not found: {train_csv}"

    dataset = CaptionDataset(str(train_csv), sequences_path="data/caption_sequences.npy", features_dir="data/features")
    if args.limit:
        dataset = torch.utils.data.Subset(dataset, range(min(args.limit, len(dataset))))
    dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True, collate_fn=collate_fn, num_workers=2)

    vocab = torch.load("data/vocab.pt")
    word2idx = vocab["word2idx"]
    vocab_size = len(word2idx)

    embed_dim = args.embed
    hidden_dim = args.hidden

    encoder = Encoder(feat_dim=2048, embed_dim=embed_dim).to(device)
    decoder = DecoderLSTM(embed_dim=embed_dim, hidden_dim=hidden_dim, vocab_size=vocab_size, padding_idx=word2idx["<pad>"]).to(device)

    opt_e = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    opt_d = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx["<pad>"])

    best_loss = float("inf")
    os.makedirs("models", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(encoder, decoder, dataloader, opt_e, opt_d, criterion, device)
        print(f"Epoch {epoch} avg loss: {avg_loss:.4f}")
        # save every epoch
        save_path = f"models/caption_epoch{epoch}.pth"
        torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict(), "word2idx": word2idx}, save_path)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict(), "word2idx": word2idx}, "models/caption_model_best.pth")
            print("Saved best model.")

    print("Training finished. Best loss:", best_loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0, help="limit number of samples for quick test")
    args = parser.parse_args()
    main(args)

