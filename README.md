# 🛰️ SatQuery AI

**SatQuery AI** is an agentic Vision-Language Assistant for remote sensing and satellite imagery analysis.

Upload satellite imagery and ask questions in natural language. The system validates the imagery, detects modality, routes the query to an appropriate analysis workflow, executes the required model pipeline, and returns an interpretable result.

## Features

- Natural-language satellite image analysis
- Agentic query routing
- Vision-Language Model support
- Visual Question Answering (VQA)
- Image captioning
- Bi-temporal change detection
- Change VQA
- Optical + SAR analysis
- PNG, JPG/JPEG, TIFF/GeoTIFF support
- Geospatial metadata and input validation
- Evidence and execution traces
- FastAPI backend
- React + Vite frontend
- Qwen2.5-VL + LoRA/PEFT support

## Architecture

```text
User
 │
 ▼
React / Vite Frontend
 │ HTTP
 ▼
FastAPI Backend
 │
 ├── Input Validation
 ├── Metadata Extraction
 ├── Modality Detection
 ├── Query Routing
 │
 ▼
Agent / Controller
 │
 ├── VQA
 ├── Captioning
 ├── Change Detection
 ├── Change VQA
 └── Optical + SAR
 │
 ▼
Qwen2.5-VL + SatQuery LoRA
 │
 ▼
Answer + Evidence + Metadata
```

## Project Structure

```text
SatQueryAI/
├── Backend/
│   ├── agent/
│   ├── preprocessing/
│   ├── models/
│   ├── scripts/
│   ├── training/
│   ├── data/
│   ├── outputs/
│   ├── hf_cache/
│   ├── .streamlit/
│   │   └── config.toml
│   ├── app.py
│   ├── main.py
│   ├── config.py
│   ├── requirements.txt
│   └── .env.example
│
├── Frontend/
│   ├── public/
│   ├── src/
│   ├── package.json
│   ├── vite.config.js
│   └── index.html
│
├── .gitignore
├── .env.example
└── README.md
```

## Supported Analysis

| Task | Purpose |
|---|---|
| VQA | Answer questions about a satellite image |
| Captioning | Generate a semantic image description |
| Change Detection | Identify differences between two images |
| Change VQA | Answer questions about temporal changes |
| Optical + SAR | Analyze complementary optical and SAR imagery |

## Supported Inputs

- PNG
- JPG / JPEG
- TIFF / GeoTIFF

For GeoTIFF files, the backend can inspect dimensions, bands, data type, CRS, georeferencing and spatial resolution.

## AI Model

The shared VLM workflow is based on:

**Qwen2.5-VL-3B-Instruct + SatQuery LoRA/PEFT adapter**

Model weights and Hugging Face caches should not be committed to Git because of their size.

## Installation

### 1. Clone

```bash
git clone https://github.com/sakshamengineer/SatQuery-AI.git
cd SatQuery-AI
```

### 2. Backend

```bash
cd Backend
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Then:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. GPU

Verify PyTorch:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

For NVIDIA GPU inference, install a CUDA-enabled PyTorch build appropriate for your system using the official PyTorch installation instructions.

Then verify:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

### 4. Hugging Face

Create a local `.env` file when authentication is needed:

```env
HF_TOKEN=your_token_here
```

Never commit tokens or credentials.

## Run Backend

From `Backend/`:

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

API:

```text
http://127.0.0.1:8000
```

Health:

```text
http://127.0.0.1:8000/health
```

API docs:

```text
http://127.0.0.1:8000/docs
```

## Run Frontend

In another terminal:

```bash
cd Frontend
npm install
npm run dev
```

The Vite frontend normally runs at:

```text
http://localhost:5173
```

## Streamlit

For the Streamlit interface:

```bash
cd Backend
python -m streamlit run app.py
```

The Streamlit configuration is stored in:

```text
Backend/.streamlit/config.toml
```

## Environment Variables

Root or backend `.env`:

```env
HF_TOKEN=
```

Frontend `.env` example:

```env
VITE_API_URL=http://127.0.0.1:8000
```

Do not commit `.env` files.

## Development

Recommended workflow:

1. Activate the backend virtual environment.
2. Start FastAPI.
3. Start the Vite frontend.
4. Test API endpoints.
5. Test image upload and validation.
6. Test query routing.
7. Test model inference.
8. Check generated evidence and outputs.

## Tech Stack

**Backend:** Python, FastAPI, Uvicorn, PyTorch, Transformers, PEFT/LoRA, Qwen2.5-VL, Rasterio, GeoPandas, OpenCV, NumPy, Pandas, Scikit-learn.

**Frontend:** React, Vite, JavaScript, HTML, CSS.

**Remote Sensing:** GeoTIFF, optical imagery, SAR imagery, metadata validation, change detection and multimodal image analysis.

## Git Hygiene

The repository should exclude:

```text
.venv/
node_modules/
hf_cache/
data/
outputs/
uploads/
.env
model weights
Python caches
frontend build artifacts
```

Large model files should be stored using an appropriate model/artifact hosting service rather than committed directly to Git.

## Contributing

1. Create a feature branch.
2. Make and test your changes.
3. Commit the changes.
4. Push the branch.
5. Open a pull request.

Example:

```bash
git checkout -b feature/new-analysis
git add .
git commit -m "Add new analysis workflow"
git push origin feature/new-analysis
```

## License

This project is intended for educational, research and demonstration purposes. Add a `LICENSE` file if you want to define formal reuse terms.

## Project

**SatQuery AI**  
Agentic Vision-Language Assistant for Remote Sensing

GitHub: https://github.com/sakshamengineer/SatQuery-AI
