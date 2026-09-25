# KrishiSaarthi AI

KrishiSaarthi AI is a crop leaf disease detection app. Upload a leaf photo to receive a model prediction, confidence score, Grad-CAM visual explanation, and disease guidance such as symptoms, favorable conditions, precautions, management, and prevention.

## Features

- Upload a JPG, JPEG, or PNG leaf image.
- Predict a disease using an EfficientNetB0 model.
- View the prediction confidence and Grad-CAM explanation.
- Read crop specific disease information from a local knowledge base.
- Use the responsive React interface on desktop and mobile.

## Project structure

```text
.
├── krishisaarthi-backend/
│   ├── main.py
│   ├── class_names.json
│   ├── disease_knowledge.json
│   └── disease_efficientnet.keras  # Add the trained model locally
└── krishisaarthi-frontend/
    ├── src/
    ├── package.json
    └── vite.config.js
```

## Requirements

- Python 3.11 
- Node.js and npm
- The trained `disease_efficientnet.keras` model file

## Run locally

Open two terminals from the project root.

### 1. Start the backend

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the backend dependencies and launch the API:

```bash
python -m pip install fastapi "uvicorn[standard]" python-multipart tensorflow numpy pillow
cd krishisaarthi-backend
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

The API starts at `http://127.0.0.1:8000`. Its interactive documentation is available at `http://127.0.0.1:8000/docs`.

### 2. Start the frontend

In a second terminal:

```bash
cd krishisaarthi-frontend
npm install
npm run dev
```

Open the local URL printed by Vite, usually `http://localhost:5173`. The frontend sends image predictions to `http://127.0.0.1:8000/predict`.

## API

### `GET /`

Returns a message confirming that the backend is running.

### `POST /predict`

Accepts an image as multipart form data under the `file` field. The JSON response includes:

- `disease`: predicted model class
- `confidence`: prediction confidence from 0 to 1
- `gradcam`: base64 encoded PNG visualization
- `reason` and `advice`: model specific explanation and advice
- `knowledge`: crop, disease cause, symptoms, favorable conditions, precautions, management, and prevention when available

## Troubleshooting

- **Backend cannot find the model:** Confirm `disease_efficientnet.keras` is in `krishisaarthi-backend/`.
- **Frontend cannot connect:** Start the API on port 8000 and open the frontend on `localhost:5173` or `127.0.0.1:5173`, which are the configured CORS origins.
- **Image upload fails:** Use a valid JPG, JPEG, or PNG image.

## Notes

Predictions are informational and should be confirmed with a local agricultural expert before taking action. Disease guidance is loaded from `disease_knowledge.json` and may not cover every region or growing condition.
