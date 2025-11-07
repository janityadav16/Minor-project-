# Object Detection - Image Captioning App

A Streamlit-based web application for generating captions from images using a deep learning model.

## Features

- 🎨 **Modern Dark UI**: Futuristic design with light blue accents
- 📷 **Camera Support**: Capture images directly from your webcam
- 📁 **File Upload**: Upload images from your device
- 🤖 **AI-Powered**: Generates captions using a trained LSTM-based model

## Setup

### 1. Install Dependencies

```bash
# Activate virtual environment
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Verify Model Files

Ensure you have:
- `models/caption_model_best.pth` - Trained model checkpoint
- `data/vocab.pt` - Vocabulary file

## Running the Application

### Option 1: Run Both Services Together (Recommended)

```bash
bash run_app.sh
```

This will:
- Start the FastAPI backend on `http://localhost:8000`
- Start the Streamlit frontend on `http://localhost:8501`

### Option 2: Run Services Separately

**Terminal 1 - Backend:**
```bash
source .venv/bin/activate
python -m uvicorn src.server:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - Frontend:**
```bash
source .venv/bin/activate
streamlit run src/streamlit_app.py --server.port 8501
```

## Usage

1. Open your browser and navigate to `http://localhost:8501`
2. Choose an input method:
   - **Camera**: Click the camera button to capture an image
   - **Upload**: Click the upload area to select an image file
3. Click **PROCESS** to generate a caption
4. View the generated caption below
5. Click **RESET** to clear the current image and start over

## Architecture

- **Backend**: FastAPI server (`src/server.py`) that handles image processing and caption generation
- **Frontend**: Streamlit app (`src/streamlit_app.py`) with modern UI
- **Model**: LSTM-based encoder-decoder architecture for caption generation

## API Endpoints

- `GET /health` - Health check endpoint
- `POST /predict` - Generate caption from uploaded image

## Troubleshooting

- **Backend not running**: Make sure the backend server is running on port 8000
- **Model files missing**: Ensure `models/caption_model_best.pth` and `data/vocab.pt` exist
- **Port conflicts**: Change ports in the run commands if 8000 or 8501 are in use

