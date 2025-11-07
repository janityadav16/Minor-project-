# src/evaluate_model.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import argparse
from pathlib import Path
from train_model import Encoder, DecoderLSTM
import numpy as np

# -------------------------
# 1. Feature Extractor (ResNet50)
# -------------------------
def extract_features(image_path, device):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = nn.Sequential(*list(model.children())[:-1])  # remove final FC layer
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(image_tensor)
    features = features.squeeze().cpu().numpy()
    return torch.tensor(features).unsqueeze(0).to(device)  # (1, 2048)

# -------------------------
# 2. Generate Caption
# -------------------------
def generate_caption(encoder, decoder, vocab, features, device, max_len=20):
    word2idx = vocab["word2idx"]
    idx2word = {idx: word for word, idx in word2idx.items()}

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        feat_emb = encoder(features)
        inputs = torch.tensor([[word2idx["<start>"]]], device=device)
        hidden = None
        caption_ids = []

        for _ in range(max_len):
            embeddings = decoder.embed(inputs)
            if hidden is None:
                lstm_input = torch.cat([feat_emb.unsqueeze(1), embeddings], dim=1)
            else:
                lstm_input = embeddings

            out, hidden = decoder.lstm(lstm_input, hidden)
            logits = decoder.fc(out[:, -1, :])  # get last output
            predicted = logits.argmax(1)
            predicted_id = predicted.item()
            caption_ids.append(predicted_id)
            if idx2word[predicted_id] == "<end>":
                break
            inputs = predicted.unsqueeze(0)

    caption = [idx2word[idx] for idx in caption_ids if idx2word[idx] not in ("<start>", "<end>", "<pad>")]
    return " ".join(caption)

# -------------------------
# 3. Main
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to test image")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load("models/caption_model_best.pth", map_location=device)
    vocab = torch.load("data/vocab.pt")

    embed_dim = 512
    hidden_dim = 512
    vocab_size = len(vocab["word2idx"])

    encoder = Encoder(feat_dim=2048, embed_dim=embed_dim).to(device)
    decoder = DecoderLSTM(embed_dim=embed_dim, hidden_dim=hidden_dim, vocab_size=vocab_size,
                          padding_idx=vocab["word2idx"]["<pad>"]).to(device)

    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

    features = extract_features(args.image, device)
    caption = generate_caption(encoder, decoder, vocab, features, device)

    print(f"\n🖼️ Caption: {caption}\n")
