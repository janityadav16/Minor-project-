import os
from flask import Flask, render_template, request
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from train_model import Encoder, DecoderLSTM

# -----------------------------
# Flask app setup
# -----------------------------
app = Flask(__name__, template_folder="../templates", static_folder="../static")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Paths
# -----------------------------
model_path = "../final_model/final-model.pth"

# -----------------------------
# Load vocab and model
# -----------------------------
print("Loading model checkpoint...")
checkpoint = torch.load(model_path, map_location=device)

word2idx = checkpoint["word2idx"]
idx2word = {idx: word for word, idx in word2idx.items()}
vocab_size = len(word2idx)

# Use same dimensions as in training
embed_dim = 512
hidden_dim = 512

# Load trained models
encoder = Encoder(feat_dim=2048, embed_dim=embed_dim).to(device)
decoder = DecoderLSTM(embed_dim=embed_dim, hidden_dim=hidden_dim, vocab_size=vocab_size).to(device)

encoder.load_state_dict(checkpoint["encoder"])
decoder.load_state_dict(checkpoint["decoder"])

encoder.eval()
decoder.eval()

# -----------------------------
# Define ResNet-50 feature extractor
# -----------------------------
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
modules = list(resnet.children())[:-1]  # remove last fc layer
resnet = nn.Sequential(*modules)
resnet.eval().to(device)

# -----------------------------
# Image Transform (same as training)
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

# -----------------------------
# Feature Extraction
# -----------------------------
def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = resnet(image)
    features = features.view(features.size(0), -1)  # (1, 2048)
    return features

# -----------------------------
# Caption Generation
# -----------------------------
def generate_caption(image_path, max_len=20):
    features = extract_features(image_path)
    with torch.no_grad():
        features_emb = encoder(features)

        sampled_ids = []
        inputs = torch.tensor([[word2idx["<start>"]]]).to(device)
        states = None

        for _ in range(max_len):
            embeddings = decoder.embed(inputs)
            lstm_input = torch.cat((features_emb.unsqueeze(1), embeddings), dim=1)
            hiddens, states = decoder.lstm(lstm_input, states)
            # use last time-step hidden state for prediction
            last_hidden = hiddens[:, -1, :]
            outputs = decoder.fc(last_hidden)
            predicted = outputs.argmax(dim=1)
            word_id = predicted.item()
            sampled_ids.append(word_id)
            if idx2word[word_id] == "<end>":
                break
            inputs = predicted.unsqueeze(0).unsqueeze(0)

    caption = [idx2word[i] for i in sampled_ids if i in idx2word]
    caption = " ".join([word for word in caption if word not in ["<start>", "<end>", "<pad>"]])
    return caption

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    caption = None
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400
        image = request.files["file"]
        if image.filename == "":
            return "No selected file", 400

        upload_dir = "../static/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, image.filename)
        image.save(filepath)

        caption = generate_caption(filepath)
        # Build URL for displaying the uploaded image
        from flask import url_for
        image_url = url_for('static', filename=f"uploads/{image.filename}")
        return render_template("index.html", caption=caption, image_url=image_url)

    return render_template("index.html", caption=None)

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
