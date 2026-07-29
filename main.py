"""
ECG Intelligence — hybrid analysis backend.

Pipeline:
  1) ecg_core.analyze_ecg_image()  -> REAL signal processing (OpenCV + scipy).
     Measures BPM and rhythm regularity directly from the image pixels.
     This part has nothing to do with any AI API — it is deterministic DSP.
  2) Gemini (vision) -> takes the image + the ALREADY-MEASURED numbers and
     gives a short clinical-style interpretation (rhythm label, note).
     Gemini is NOT asked to invent the BPM — only to interpret what the
     measurement + image show. This is the answer to "why not just use
     Gemini directly": Gemini never sees the job of measuring, only of
     reading the result like a clinician would.

Env vars required (set these in Render -> your service -> Environment):
  GEMINI_API_KEY = <your key from https://aistudio.google.com/apikey>
"""
import os
import json
import traceback

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ecg_core import analyze_ecg_image

import google.generativeai as genai

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="ECG Intelligence Backend")

# Allow your frontend (Netlify/Vercel/local file) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten to your real frontend domain before the final submission
    allow_methods=["*"],
    allow_headers=["*"],
)


def interpret_with_gemini(image_bytes: bytes, measured: dict) -> dict:
    """Ask Gemini to interpret the ALREADY-MEASURED signal, not to remeasure it."""
    if not GEMINI_API_KEY:
        # No key configured yet -> fall back to a rule-based label so the
        # demo still works end-to-end while you're setting up the key.
        return _rule_based_label(measured)

    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""Siz klinik yordamchi AI'siz. Quyida EKG rasmidan bizning signal-processing
tizimimiz o'lchagan haqiqiy ma'lumotlar berilgan (siz bu raqamlarni QAYTA HISOBLAMANG,
faqat sharhlang):

- O'lchangan yurak tezligi: {measured['bpm']} BPM
- Aniqlangan yurak siklllari soni: {measured['n_beats_detected']}
- RR-interval muntazamligi (variatsiya koeffitsienti): {measured['regularity_cv']}
- Ritm muntazammi: {"Ha" if measured['is_regular'] else "Yo'q, sezilarli o'zgaruvchan"}

Rasmga ham qarab (grid qog'ozidagi P-QRS-T shakllarini tekshiring), FAQAT quyidagi JSON
formatida javob bering, boshqa hech qanday matn qo'shmang:

{{"rhythm": "<qisqa klinik nom, masalan: Sinus Ritm / Sinus Bradikardiyasi / Sinus Taxikardiyasi / Atrial Fibrillyatsiya shubhasi / Noaniq>",
  "note": "<1-2 gapli qisqa klinik izoh, o'zbek tilida>",
  "confidence": "<past/o'rta/yuqori>"}}"""

    try:
        response = model.generate_content(
            [prompt, {"mime_type": "image/jpeg", "data": image_bytes}],
            generation_config={"response_mime_type": "application/json"},
        )
        parsed = json.loads(response.text)
        return parsed
    except Exception:
        traceback.print_exc()
        return _rule_based_label(measured)


def _rule_based_label(measured: dict) -> dict:
    """Zero-dependency fallback so the demo never hard-fails if the AI call
    errors out (rate limit, no key yet, network hiccup, etc.)."""
    bpm = measured["bpm"]
    if not measured["is_regular"]:
        return {"rhythm": "Aritmiya shubhasi", "note": "RR-intervallar sezilarli o'zgaruvchan.", "confidence": "past"}
    if bpm < 60:
        return {"rhythm": "Sinus Bradikardiyasi", "note": f"O'rtacha {bpm} BPM, me'yordan past.", "confidence": "o'rta"}
    if bpm > 100:
        return {"rhythm": "Sinus Taxikardiyasi", "note": f"O'rtacha {bpm} BPM, me'yordan yuqori.", "confidence": "o'rta"}
    return {"rhythm": "Sinus Ritm", "note": f"O'rtacha {bpm} BPM, me'yor doirasida.", "confidence": "o'rta"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    image_bytes = await file.read()

    try:
        measured = analyze_ecg_image(image_bytes)
    except ValueError as e:
        return JSONResponse(status_code=422, content={"error": str(e)})
    except Exception:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": "Rasmni tahlil qilishda kutilmagan xato."})

    interpretation = interpret_with_gemini(image_bytes, measured)

    return {
        "heart_rate": measured["bpm"],
        "rhythm": interpretation.get("rhythm", "Noaniq"),
        "note": interpretation.get("note", ""),
        "confidence": interpretation.get("confidence", ""),
        "regularity_cv": measured["regularity_cv"],
        "beats_detected": measured["n_beats_detected"],
    }


@app.get("/")
async def health():
    return {"status": "ok", "service": "ECG Intelligence backend"}
