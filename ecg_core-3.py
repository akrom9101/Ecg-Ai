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


def _count_grid_lines(gray_eq, h, w):
    """Count long, straight, evenly-oriented lines via Hough transform and
    split them into near-horizontal vs near-vertical. Real ECG graph paper
    has dozens of each, evenly spaced, spanning most of the strip. A photo
    of a person, document, or object essentially never does — a shirt fold
    or stethoscope tube gives at most a couple of short, angled segments.
    """
    edges = cv2.Canny(gray_eq, 20, 60)
    min_len = int(min(h, w) * 0.35)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=60,
        minLineLength=min_len, maxLineGap=6,
    )
    if lines is None:
        return 0, 0
    horiz, vert = 0, 0
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 5 or angle > 175:
            horiz += 1
        elif 85 < angle < 95:
            vert += 1
    return horiz, vert


def _validate_ecg_layout(img):
    """Reject ordinary photos before signal extraction.

    Paper ECGs have a fine millimetre grid: dozens of long, straight,
    evenly-spaced lines running both horizontally and vertically across
    most of the strip. This geometric signature — not color — is what
    actually tells real ECG paper apart from ordinary photos. Color (the
    grid is usually red/pink) turned out to be an unreliable pre-filter:
    real phone photos are frequently washed out by lighting/JPEG
    compression to the point of failing any color threshold, which
    produces false rejections of genuine ECG images. So color is no
    longer a hard gate — only the counted geometry below is.
    """
    b, g, r = [channel.astype(np.int16) for channel in cv2.split(img)]

    # Boost local contrast first (CLAHE) so a faint, low-contrast grid/trace
    # from a phone photo or heavy JPEG compression reads the same as a crisp
    # scan would — used consistently by both checks below.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    horiz_lines, vert_lines = _count_grid_lines(gray_eq, h, w)
    # A real grid strip has many lines in BOTH directions, evenly spanning
    # the strip. Requiring both counts (not just one) rejects photos that
    # happen to have some parallel edges in only one orientation (a
    # doorframe, a table edge, window blinds).
    if horiz_lines < 12 or vert_lines < 12:
        raise ValueError(
            "ECG rasmi tasdiqlanmadi — muntazam EKG grid chiziqlari topilmadi. "
            "Oddiy rasm yoki grid ko'rinmaydigan surat qabul qilinmaydi."
        )

    # Require dark ink running across a solid majority of the width — a real
    # ECG trace spans the whole strip; scattered shadows/hair/objects in an
    # ordinary photo only cover a patch of columns. Uses the same
    # contrast-boosted image as the line check so faint, low-contrast photos
    # are judged consistently rather than by raw (washed-out) pixel values.
    dark_neutral = (gray_eq < 150) & (np.abs(r - g) < 40) & (np.abs(g - b) < 40)
    trace_columns = float(dark_neutral.any(axis=0).mean())
    if trace_columns < 0.4:
        raise ValueError(
            "ECG chizig'i yetarli ko'rinmadi — rasmni tekisroq va yorug'roq oling."
        )

    return {
        "grid_lines_h": horiz_lines,
        "grid_lines_v": vert_lines,
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
