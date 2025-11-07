import io
import os
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

from train_model import Encoder, DecoderLSTM


app = FastAPI(title="Object Caption API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CaptionService:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_feature_extractor()
        self._load_checkpoint()

    def _init_feature_extractor(self) -> None:
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1]).to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def _load_checkpoint(self) -> None:
        ckpt_path = "models/caption_model_best.pth"
        vocab_path = "data/vocab.pt"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model checkpoint not found at {ckpt_path}")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocab file not found at {vocab_path}")

        self.checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.vocab = torch.load(vocab_path)
        word2idx = self.vocab["word2idx"]
        vocab_size = len(word2idx)

        self.encoder = Encoder(feat_dim=2048, embed_dim=512).to(self.device)
        self.decoder = DecoderLSTM(embed_dim=512, hidden_dim=512, vocab_size=vocab_size,
                                   padding_idx=word2idx["<pad>"]).to(self.device)
        self.encoder.load_state_dict(self.checkpoint["encoder"])
        self.decoder.load_state_dict(self.checkpoint["decoder"])
        self.encoder.eval()
        self.decoder.eval()

        self.idx2word = {idx: word for word, idx in word2idx.items()}

    def _extract_features(self, image: Image.Image) -> torch.Tensor:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.feature_extractor(tensor)
        feats = feats.squeeze().unsqueeze(0)  # (1, 2048)
        return feats

    def generate_caption(self, image: Image.Image, max_len: int = 20) -> str:
        features = self._extract_features(image)
        with torch.no_grad():
            feat_emb = self.encoder(features)
            word2idx = self.vocab["word2idx"]
            inputs = torch.tensor([[word2idx["<start>"]]], device=self.device)
            hidden = None
            caption_ids = []
            for _ in range(max_len):
                embeddings = self.decoder.embed(inputs)
                if hidden is None:
                    lstm_input = torch.cat([feat_emb.unsqueeze(1), embeddings], dim=1)
                else:
                    lstm_input = embeddings
                out, hidden = self.decoder.lstm(lstm_input, hidden)
                logits = self.decoder.fc(out[:, -1, :])
                predicted = logits.argmax(1)
                predicted_id = predicted.item()
                caption_ids.append(predicted_id)
                token = self.idx2word[predicted_id]
                if token == "<end>":
                    break
                inputs = predicted.unsqueeze(0)
        words = [self.idx2word[idx] for idx in caption_ids if self.idx2word[idx] not in ("<start>", "<end>", "<pad>")]
        return " ".join(words)


service: CaptionService | None = None


@app.on_event("startup")
def _startup() -> None:
    global service
    try:
        service = CaptionService()
    except Exception as e:
        # Defer hard failure until first request to return helpful error JSON
        service = None
        print(f"Startup warning: {e}")


@app.get("/health")
def health() -> dict:
    # Lightweight health check that also verifies model files exist
    ckpt_ok = os.path.exists("models/caption_model_best.pth")
    vocab_ok = os.path.exists("data/vocab.pt")
    return {"status": "ok", "checkpoint": ckpt_ok, "vocab": vocab_ok}


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> Dict[str, Any]:
    global service
    try:
        if service is None:
            service = CaptionService()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model initialization failed: {e}")

    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image upload")

    try:
        caption = service.generate_caption(img)
        # Keep contract with frontend: respond with 'boxes' array even if empty
        return {"caption": caption, "boxes": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")


# Convenience for `python src/server.py` during local dev
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=True)


