"""
Core ECG signal-extraction pipeline.
Takes a photo/scan of a paper ECG strip and returns a measured BPM +
rhythm-regularity metric using real signal processing (no AI/LLM here —
this module is the deterministic "instrument" layer).
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


def _periodicity(profile, min_lag=3, max_lag=60):
    """Return the strongest normalized repeating pattern in a 1-D profile."""
    profile = np.asarray(profile, dtype=np.float32)
    if profile.size < 20:
        return 0.0
    # Keep autocorrelation bounded for phone photos with very large dimensions.
    stride = max(1, int(profile.size / 1800))
    profile = profile[::stride]
    profile = profile - np.median(profile)
    energy = float(np.dot(profile, profile))
    if energy <= 1e-6:
        return 0.0
    ac = np.correlate(profile, profile, mode="full")[len(profile) - 1:] / energy
    lo = min_lag
    hi = min(max_lag, len(ac) - 1)
    return float(np.max(ac[lo:hi + 1])) if hi >= lo else 0.0


def _validate_ecg_layout(img):
    """Reject ordinary photos before signal extraction.

    The previous pipeline only needed two dark peaks. A portrait, document, or
    random dark object can satisfy that condition. Paper ECGs supported by this
    project have a repeated red/pink millimetre grid plus a dark trace, so both
    characteristics are required here.
    """
    b, g, r = [channel.astype(np.int16) for channel in cv2.split(img)]
    redness = r - ((g + b) / 2.0)
    # ECG paper grid is commonly red/pink. The saturation floor prevents
    # neutral grey backgrounds — and warm-toned ordinary photos (skin, wood,
    # sunsets) — from being treated as grid lines.
    grid_mask = (r > 100) & (redness > 18) & (r > g + 12)
    grid_fraction = float(grid_mask.mean())
    if grid_fraction < 0.02:
        raise ValueError(
            "ECG rasmi tasdiqlanmadi — qizil/pushti grid topilmadi. "
            "Faqat gridli EKG rasmini yuboring."
        )

    col_profile = grid_mask.mean(axis=0)
    row_profile = grid_mask.mean(axis=1)
    periodicity = max(_periodicity(col_profile), _periodicity(row_profile))
    line_strength = max(float(col_profile.max()), float(row_profile.max()))
    # Both signals must independently look like a printed grid now — a random
    # repeating texture (fabric, tiles, blinds) rarely satisfies both at once.
    if periodicity < 0.15 or line_strength < 0.12:
        raise ValueError(
            "ECG rasmi tasdiqlanmadi — muntazam EKG grid chiziqlari topilmadi. "
            "Oddiy rasm yoki grid ko'rinmaydigan surat qabul qilinmaydi."
        )

    # Require dark, mostly-neutral ink running across MOST of the width —
    # a real ECG trace spans the whole strip; scattered shadows/hair/objects
    # in an ordinary photo only cover a patch of columns.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark_neutral = (gray < 145) & (np.abs(r - g) < 34) & (np.abs(g - b) < 34)
    trace_columns = float(dark_neutral.any(axis=0).mean())
    if trace_columns < 0.55:
        raise ValueError(
            "ECG chizig'i yetarli ko'rinmadi — rasmni tekisroq va yorug'roq oling."
        )

    return {
        "grid_fraction": round(grid_fraction, 4),
        "grid_periodicity": round(periodicity, 3),
        "trace_columns": round(trace_columns, 3),
    }


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


def _extract_trace_y(img):
    """Isolate the ink trace from the red/pink grid and return one y-position
    per x-column (the trace's vertical position across time)."""
    b, g, r = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
    # grid is reddish (R high, B/G lower); trace ink is dark & roughly neutral (low R too)
    is_dark = (r < 150) & (g < 150) & (b < 150)
    redness = r - (g + b) / 2
    is_trace = is_dark & (redness < 40)

    H, W = is_dark.shape
    ys = np.full(W, np.nan)
    for x in range(W):
        col = np.where(is_trace[:, x])[0]
        if len(col):
            ys[x] = col.mean()

    # fill small gaps by linear interpolation
    valid = ~np.isnan(ys)
    if valid.sum() < W * 0.3:
        raise ValueError("EKG chizig'i aniqlanmadi — rasm sifati past yoki qog'oz to'liq ko'rinmayapti")
    idx = np.arange(W)
    ys = np.interp(idx, idx[valid], ys[valid])
    return ys


def _analyze_oriented(img):
    """Run the full measurement pipeline assuming the trace runs
    left-to-right in `img` as given (no rotation handling here)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    px_per_mm = _estimate_px_per_mm(gray)

    ys = _extract_trace_y(img)
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
        raise ValueError(
            f"Yurak tezligi ishonchli diapazonda aniqlanmadi ({bpm:.1f} BPM)"
        )
    regularity_cv = float(np.std(rr_seconds) / np.mean(rr_seconds))
    is_regular = regularity_cv < 0.10

    # mean peak "sharpness" (prominence) as a rough confidence signal, used
    # only to pick the best orientation when trying several rotations.
    mean_prominence = float(np.mean(props["prominences"])) if len(props.get("prominences", [])) else 0.0

    return {
        "bpm": round(bpm, 1),
        "n_beats_detected": int(len(peaks)),
        "rr_intervals_ms": [round(float(s) * 1000, 1) for s in rr_seconds],
        "regularity_cv": round(regularity_cv, 3),
        "is_regular": bool(is_regular),
        "px_per_mm_estimated": round(float(px_per_mm), 2),
        "_confidence": int(len(peaks)) * mean_prominence,
    }


def analyze_ecg_image(path_or_bytes):
    img = _load_image(path_or_bytes)
    _validate_ecg_layout(img)

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


if __name__ == "__main__":
    import sys
    result = analyze_ecg_image(sys.argv[1] if len(sys.argv) > 1 else "/home/claude/test_ecg.png")
    print(result)