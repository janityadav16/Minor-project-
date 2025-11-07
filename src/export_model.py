import torch
import os
from train_model import Encoder, DecoderLSTM

def export_model():
    # Paths
    checkpoint_path = "models/caption_model_best.pth"
    export_dir = "final_model"
    export_path = os.path.join(export_dir, "final-model.pth")

    # Create directory if it doesn't exist
    os.makedirs(export_dir, exist_ok=True)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Load encoder and decoder structures
    encoder = Encoder(feat_dim=2048, embed_dim=512)
    decoder = DecoderLSTM(embed_dim=512, hidden_dim=512, vocab_size=len(checkpoint["word2idx"]))

    # Load weights
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

    # Combine everything into one file
    export_bundle = {
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "word2idx": checkpoint["word2idx"],
        "config": {
            "feat_dim": 2048,
            "embed_dim": 512,
            "hidden_dim": 512
        }
    }

    # Save the final model
    torch.save(export_bundle, export_path)
    print(f"✅ Final model saved to: {export_path}")

if __name__ == "__main__":
    export_model()
