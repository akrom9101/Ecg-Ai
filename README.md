# ECG Intelligence

FastAPI web application for analyzing grid-based ECG images. The
signal-processing layer measures BPM and rhythm regularity; Gemini optionally
provides a short interpretation. If `GEMINI_API_KEY` is absent, the backend
uses a rule-based fallback.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"  # optional
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open the interface at `http://localhost:8000/`.

Health check: `GET http://localhost:8000/health`

ECG analysis: `POST http://localhost:8000/analyze` with a multipart image field
named `file` and an optional `language` field (`uz`, `en`, or `ru`).

The web interface is served by the same backend at
`http://localhost:8000/`. Do not open `index.html` with a file browser unless
the backend is already running; serving it through FastAPI avoids browser
CORS and relative-URL issues.

## Deploy on Render

Create a **Web Service** from this GitHub repository:

- **Runtime:** Python
- **Build command:** `pip install -r requirements.txt`
- **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Environment variable:** `GEMINI_API_KEY` (optional; add it in Render Environment)

After deployment, open the Render URL. The frontend and `/analyze` API are
served from the same URL.

This service is an analysis aid, not a medical diagnosis. Confirm results with
a qualified clinician.