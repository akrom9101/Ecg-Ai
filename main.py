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
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from ecg_core import analyze_ecg_image

import google.generativeai as genai

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg"}
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="ECG Intelligence Backend")
BASE_DIR = Path(__file__).resolve().parent

# Allow your frontend (Netlify/Vercel/local file) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten to your real frontend domain before the final submission
    allow_methods=["*"],
    allow_headers=["*"],
)


def _language_name(language: str) -> str:
    return {"uz": "o'zbek", "en": "English", "ru": "русском"}.get(language, "o'zbek")


def interpret_with_gemini(
    image_bytes: bytes, measured: dict, language: str = "uz", mime_type: str = "image/jpeg"
) -> dict:
    """Ask Gemini to interpret the ALREADY-MEASURED signal, not to remeasure it."""
    if not GEMINI_API_KEY:
        # No key configured yet -> fall back to a rule-based label so the
        # demo still works end-to-end while you're setting up the key.
        return _rule_based_label(measured, language)

    model = genai.GenerativeModel("gemini-2.5-flash")
    response_language = _language_name(language)
    prompt = f"""Siz klinik yordamchi AI'siz. Quyida EKG rasmidan bizning signal-processing
tizimimiz o'lchagan haqiqiy ma'lumotlar berilgan (siz bu raqamlarni QAYTA HISOBLAMANG,
faqat sharhlang):

- O'lchangan yurak tezligi: {measured['bpm']} BPM
- Aniqlangan yurak siklllari soni: {measured['n_beats_detected']}
- RR-interval muntazamligi (variatsiya koeffitsienti): {measured['regularity_cv']}
- Ritm muntazammi: {"Ha" if measured['is_regular'] else "Yo'q, sezilarli o'zgaruvchan"}

Avval rasmning o'zi HAQIQIY qog'ozga bosilgan yoki ekrandagi EKG/kardiogramma
strip'i ekanligini tasdiqlang (P-QRS-T shakllari, millimetrli grid ko'rinishi kerak).
Agar rasm EKG bo'lmasa (odam surati, hujjat, tabiat, tasodifiy fon va h.k.), "is_valid_ecg"
maydonini false qiling va boshqa maydonlarni bo'sh/"Noaniq" qoldiring.

Agar rasm haqiqatan ham EKG bo'lsa, P-QRS-T shakllarini tekshiring, lekin
o'lchangan BPMni o'zgartirmang. Rasmda yetarli klinik belgi bo'lmasa ritmni "Noaniq"
deb qaytaring; taxmin bilan aritmiya yozmang. Javobni {response_language} tilida bering.
FAQAT quyidagi JSON formatida javob bering, boshqa matn qo'shmang:

{{"is_valid_ecg": <true yoki false>,
  "rhythm": "<qisqa klinik nom, masalan: Sinus Ritm / Sinus Bradikardiyasi / Sinus Taxikardiyasi / Atrial Fibrillyatsiya shubhasi / Noaniq>",
  "note": "<1-2 gapli qisqa klinik izoh, {response_language} tilida>",
  "confidence": "<past/o'rta/yuqori>"}}"""

    try:
        response = model.generate_content(
            [prompt, {"mime_type": mime_type, "data": image_bytes}],
            generation_config={"response_mime_type": "application/json"},
        )
        text = (response.text or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("Gemini JSON javobi object emas")
        return {
            "is_valid_ecg": bool(parsed.get("is_valid_ecg", True)),
            "rhythm": str(parsed.get("rhythm") or "Noaniq"),
            "note": str(parsed.get("note") or ""),
            "confidence": str(parsed.get("confidence") or "past"),
        }
    except Exception:
        traceback.print_exc()
        fallback = _rule_based_label(measured, language)
        fallback["is_valid_ecg"] = True  # Gemini unavailable — DSP grid-check already passed
        return fallback


def _rule_based_label(measured: dict, language: str = "uz") -> dict:
    """Zero-dependency fallback so the demo never hard-fails if the AI call
    errors out (rate limit, no key yet, network hiccup, etc.)."""
    bpm = measured["bpm"]
    if language == "en":
        if not measured["is_regular"]:
            return {"rhythm": "Possible arrhythmia", "note": "RR intervals are significantly variable.", "confidence": "low"}
        if bpm < 60:
            return {"rhythm": "Sinus bradycardia", "note": f"Average {bpm} BPM, below normal.", "confidence": "medium"}
        if bpm > 100:
            return {"rhythm": "Sinus tachycardia", "note": f"Average {bpm} BPM, above normal.", "confidence": "medium"}
        return {"rhythm": "Sinus rhythm", "note": f"Average {bpm} BPM, within normal range.", "confidence": "medium"}
    if language == "ru":
        if not measured["is_regular"]:
            return {"rhythm": "Возможная аритмия", "note": "Интервалы RR заметно изменчивы.", "confidence": "низкая"}
        if bpm < 60:
            return {"rhythm": "Синусовая брадикардия", "note": f"Средняя частота {bpm} BPM, ниже нормы.", "confidence": "средняя"}
        if bpm > 100:
            return {"rhythm": "Синусовая тахикардия", "note": f"Средняя частота {bpm} BPM, выше нормы.", "confidence": "средняя"}
        return {"rhythm": "Синусовый ритм", "note": f"Средняя частота {bpm} BPM, в пределах нормы.", "confidence": "средняя"}
    if not measured["is_regular"]:
        return {"rhythm": "Aritmiya shubhasi", "note": "RR-intervallar sezilarli o'zgaruvchan.", "confidence": "past"}
    if bpm < 60:
        return {"rhythm": "Sinus Bradikardiyasi", "note": f"O'rtacha {bpm} BPM, me'yordan past.", "confidence": "o'rta"}
    if bpm > 100:
        return {"rhythm": "Sinus Taxikardiyasi", "note": f"O'rtacha {bpm} BPM, me'yordan yuqori.", "confidence": "o'rta"}
    return {"rhythm": "Sinus Ritm", "note": f"O'rtacha {bpm} BPM, me'yor doirasida.", "confidence": "o'rta"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), language: str = Form("uz")):
    language = language if language in {"uz", "en", "ru"} else "uz"
    if file.content_type and file.content_type.lower() not in ALLOWED_IMAGE_TYPES:
        return JSONResponse(
            status_code=415,
            content={
                "error_code": "INVALID_FILE",
                "error": "Faqat PNG, JPG yoki JPEG rasm qabul qilinadi.",
            },
        )
    image_bytes = await file.read()
    if not image_bytes:
        return JSONResponse(
            status_code=422,
            content={"error_code": "INVALID_FILE", "error": "Rasm fayli bo'sh."},
        )
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            status_code=413,
            content={"error_code": "FILE_TOO_LARGE", "error": "Rasm hajmi 10 MB dan oshmasligi kerak."},
        )

    try:
        measured = analyze_ecg_image(image_bytes)
    except ValueError as e:
        return JSONResponse(
            status_code=422,
            content={"error_code": "NOT_ECG_IMAGE", "error": str(e)},
        )
    except Exception:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error_code": "ANALYZE_FAILED", "error": "Rasmni tahlil qilishda kutilmagan xato."},
        )

    mime_type = "image/png" if file.content_type == "image/png" else "image/jpeg"
    interpretation = interpret_with_gemini(image_bytes, measured, language, mime_type)

    if not interpretation.get("is_valid_ecg", True):
        return JSONResponse(
            status_code=422,
            content={
                "error_code": "NOT_ECG_IMAGE",
                "error": "Bu rasm EKG/kardiogramma emas — iltimos, gridli EKG strip rasmini yuboring.",
            },
        )

    return {
        "heart_rate": measured["bpm"],
        "rhythm": interpretation.get("rhythm", "Noaniq"),
        "note": interpretation.get("note", ""),
        "confidence": interpretation.get("confidence", ""),
        "regularity_cv": measured["regularity_cv"],
        "beats_detected": measured["n_beats_detected"],
        "language": language,
    }


@app.get("/", include_in_schema=False)
async def web_app():
    return FileResponse(BASE_DIR / "index.html", media_type="text/html")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "ECG Intelligence backend"}