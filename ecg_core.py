"""
Core ECG signal-extraction pipeline.

Simple, focused job:
  1) Find the ink trace on the strip. The paper background is pink/salmon;
     the trace is drawn in either BLACK or BLUE ink — nothing else matters.
     If a real, continuous, wiggly trace spans most of the image width,
     that alone is treated as "this is an ECG pattern". No grid-color or
     grid-geometry checks — those turned out to reject real (washed-out,
     low-contrast) photos while adding little real protection.
  2) Measure BPM + rhythm regularity from that trace (deterministic DSP,
     no AI here).
  3) Build the prompt that asks Gemini to *interpret* those already-measured
     numbers (Gemini never re-measures BPM — it only reads the trace like a
     clinician would and gives a short interpretation).
"""
import numpy as np
import cv2
from scipy.signal import find_peaks


def _load_image(path_or_bytes):
    if isinstance(path_or_bytes, (bytes, bytearray)):
        arr = np.frombuffer(path_or_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(path_or_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Rasmni o'qib bo'lmadi (unsupported/corrupt image)")
    h, w = img.shape[:2]
    if h < 240 or w < 240:
        raise ValueError("ECG rasmi juda kichik — kamida 240×240 px rasm yuboring")
    if h * w > 30_000_000:
        scale = (30_000_000 / float(h * w)) ** 0.5
        img = cv2.resize(img, (max(240, int(w * scale)), max(240, int(h * scale))), interpolation=cv2.INTER_AREA)
    return img


def _trace_mask(img):
    """Pixels that belong to the ink trace: either black or blue.

    Background (pink/salmon grid paper) has R clearly higher than B, and is
    not very dark. Trace ink is either:
      - black/dark-grey: all channels low and close to each other, or
      - blue: B channel clearly dominant over R (and not washed out).
    Either condition is enough — we don't care which color pen was used.
    """
    b, g, r = [c.astype(np.int16) for c in cv2.split(img)]

    is_black = (r < 140) & (g < 140) & (b < 140) & (np.abs(r - g) < 40) & (np.abs(g - b) < 40)
    is_blue = (b > r + 15) & (b > 60) & (r < 180)

    return is_black | is_blue


def _extract_trace_y(img):
    """Isolate the ink trace and return one y-position per x-column (the
    trace's vertical position across time). Raises if too little of the
    strip's width actually shows a trace — that's the one and only gate
    for "is this an ECG pattern": a real strip has a continuous line
    running almost the full width; an ordinary photo won't.
    """
    mask = _trace_mask(img)
    H, W = mask.shape

    ys = np.full(W, np.nan)
    for x in range(W):
        col = np.where(mask[:, x])[0]
        if len(col):
            ys[x] = col.mean()

    valid = ~np.isnan(ys)
    coverage = float(valid.mean())
    if coverage < 0.5:
        raise ValueError(
            "EKG chizig'i aniqlanmadi — qora yoki ko'k rangdagi uzluksiz chiziq topilmadi. "
            "Faqat pushti fonli EKG rasmini yuboring."
        )

    idx = np.arange(W)
    ys = np.interp(idx, idx[valid], ys[valid])
    return ys, coverage


def _estimate_px_per_mm(gray):
    """Estimate the small-grid spacing in pixels using the autocorrelation of
    column-wise darkness (grid lines create a periodic signal). We take the
    FIRST prominent peak (the fundamental), not the global max, since
    harmonics (2x, 3x the true spacing) are often taller than the fundamental."""
    row_profile = 255 - gray.mean(axis=0)
    row_profile = row_profile - row_profile.mean()
    ac = np.correlate(row_profile, row_profile, mode="full")
    ac = ac[len(ac) // 2:]
    lo, hi = 3, 40
    window = ac[lo:hi]
    peak_positions, _ = find_peaks(window, prominence=window.max() * 0.15)
    if len(peak_positions) == 0:
        return 8.0  # fallback default
    return float(lo + peak_positions[0])


def _analyze_oriented(img):
    """Run the full measurement pipeline assuming the trace runs
    left-to-right in `img` as given (no rotation handling here)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    px_per_mm = _estimate_px_per_mm(gray)

    ys, coverage = _extract_trace_y(img)
    signal = -(ys - ys.mean())  # invert: image-y grows downward, R-wave = local min in y

    k = max(3, int(px_per_mm // 3) | 1)
    kernel = np.ones(k) / k
    smooth = np.convolve(signal, kernel, mode="same")

    min_distance_px = max(int(px_per_mm * 8), 10)
    prominence = (smooth.max() - smooth.min()) * 0.35
    peaks, props = find_peaks(smooth, distance=min_distance_px, prominence=prominence)

    if len(peaks) < 3:
        raise ValueError("Yetarli R-peak topilmadi — rasmda kamida 3 ta to'liq yurak sikli bo'lishi kerak")

    rr_px = np.diff(peaks).astype(float)
    paper_speed_mm_s = 25.0
    rr_seconds = (rr_px / px_per_mm) / paper_speed_mm_s
    bpm_series = 60.0 / rr_seconds

    bpm = float(np.median(bpm_series))
    if bpm < 30 or bpm > 220:
        raise ValueError(f"Yurak tezligi ishonchli diapazonda aniqlanmadi ({bpm:.1f} BPM)")

    regularity_cv = float(np.std(rr_seconds) / np.mean(rr_seconds))
    is_regular = regularity_cv < 0.10
    mean_prominence = float(np.mean(props["prominences"])) if len(props.get("prominences", [])) else 0.0

    return {
        "bpm": round(bpm, 1),
        "n_beats_detected": int(len(peaks)),
        "rr_intervals_ms": [round(float(s) * 1000, 1) for s in rr_seconds],
        "regularity_cv": round(regularity_cv, 3),
        "is_regular": bool(is_regular),
        "px_per_mm_estimated": round(float(px_per_mm), 2),
        "trace_coverage": round(coverage, 3),
        "_confidence": int(len(peaks)) * mean_prominence,
    }


def analyze_ecg_image(path_or_bytes):
    """Load, try all 4 orientations (the trace may run top-to-bottom if the
    photo was taken sideways), and return the measurement from whichever
    orientation gave the cleanest signal."""
    img = _load_image(path_or_bytes)

    candidates = {
        0: img,
        90: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
        180: cv2.rotate(img, cv2.ROTATE_180),
        270: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }

    best_result, best_error = None, None
    for angle, candidate_img in candidates.items():
        try:
            result = _analyze_oriented(candidate_img)
            if best_result is None or result["_confidence"] > best_result["_confidence"]:
                best_result = result
        except ValueError as e:
            best_error = e

    if best_result is None:
        raise best_error  # every orientation failed — surface the last reason

    best_result.pop("_confidence", None)
    return best_result


# ---------------------------------------------------------------------------
# Gemini prompt building — lives here so the "already-measured numbers" and
# the prompt that references them never drift apart.
# ---------------------------------------------------------------------------

_LANGUAGE_NAMES = {"uz": "o'zbek", "en": "English", "ru": "русском"}


def build_gemini_prompt(measured: dict, language: str = "uz") -> str:
    """Build the prompt asking Gemini to interpret (not re-measure) the
    already-measured signal, and to confirm the image really is an ECG."""
    response_language = _LANGUAGE_NAMES.get(language, "o'zbek")
    return f"""Siz klinik yordamchi AI'siz. Quyida EKG rasmidan bizning signal-processing
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


if __name__ == "__main__":
    import sys
    result = analyze_ecg_image(sys.argv[1] if len(sys.argv) > 1 else "/home/claude/test_ecg.png")
    print(result)
    print(build_gemini_prompt(result))
