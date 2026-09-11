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
    return img


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


def analyze_ecg_image(path_or_bytes):
    img = _load_image(path_or_bytes)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    px_per_mm = _estimate_px_per_mm(gray)

    ys = _extract_trace_y(img)
    signal = -(ys - ys.mean())  # invert: image-y grows downward, R-wave = local min in y

    # smooth a touch to reduce pixel jitter before peak picking
    k = max(3, int(px_per_mm // 3) | 1)
    kernel = np.ones(k) / k
    smooth = np.convolve(signal, kernel, mode="same")

    min_distance_px = max(int(px_per_mm * 8), 10)   # refractory: no two beats closer than ~8mm (~300bpm cap)
    prominence = (smooth.max() - smooth.min()) * 0.35
    peaks, props = find_peaks(smooth, distance=min_distance_px, prominence=prominence)

    if len(peaks) < 2:
        raise ValueError("Yetarli R-peak topilmadi — rasmda kamida 2-3 to'liq yurak sikli bo'lishi kerak")

    rr_px = np.diff(peaks).astype(float)
    paper_speed_mm_s = 25.0
    rr_seconds = (rr_px / px_per_mm) / paper_speed_mm_s
    bpm_series = 60.0 / rr_seconds

    bpm = float(np.median(bpm_series))
    regularity_cv = float(np.std(rr_seconds) / np.mean(rr_seconds))  # coefficient of variation
    is_regular = regularity_cv < 0.10

    return {
        "bpm": round(bpm, 1),
        "n_beats_detected": int(len(peaks)),
        "rr_intervals_ms": [round(float(s) * 1000, 1) for s in rr_seconds],
        "regularity_cv": round(regularity_cv, 3),
        "is_regular": bool(is_regular),
        "px_per_mm_estimated": round(float(px_per_mm), 2),
    }


if __name__ == "__main__":
    import sys
    result = analyze_ecg_image(sys.argv[1] if len(sys.argv) > 1 else "/home/claude/test_ecg.png")
    print(result)
