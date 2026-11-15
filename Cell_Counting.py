
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ----------------------------- settings -----------------------------------
INPUT_FOLDER = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/yogendra/Downloads/CelI-IA_Neubauer_Dataset/Raw_Images/Neubauer_Sample_037.jpg")
OUTPUT_FOLDER = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("Processed_Results")
EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

MAX_SIDE = 1800        # working resolution (longest side)
CELLS_PER_SIDE = 4     # small squares per side of the large square (corner square = 4)
DILUTION = 1.0         # dilution factor for concentration estimate
LINE_COVER_THR = 0.30  # fraction of row/col that must be "line" to count as a grid line
BAND_MERGE_FRAC = 0.025  # lines closer than this (x image size) are merged into one band


# ----------------------------- helpers ------------------------------------
def odd(n):
    n = int(n)
    return n if n % 2 == 1 else n + 1


def load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        try:
            from PIL import Image
            arr = np.array(Image.open(path))
            img = arr[..., ::-1] if arr.ndim == 3 and arr.shape[2] >= 3 else arr
        except Exception:
            return None
    if img.dtype != np.uint8:
        img = cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    s = MAX_SIDE / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def valid_region(gray):
    """Mask of the lit field of view (removes the black vignette)."""
    blur = cv2.GaussianBlur(gray, (0, 0), 5)
    _, m = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (lab == k).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, cnts, -1, 255, -1)
    return cv2.erode(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))


def bright_binary(gray, valid):
    """Bright structures (grid lines + bright cell centres) via Otsu inside valid area."""
    g = cv2.GaussianBlur(gray, (0, 0), 2)
    vals = g[valid > 0].reshape(-1, 1)
    if vals.size == 0:
        vals = g.reshape(-1, 1)
    t, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    b = ((g > t) & (valid > 0)).astype(np.uint8) * 255
    return b


# ------------------------------ deskew -------------------------------------
def deskew(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    valid = valid_region(gray)
    binary = bright_binary(gray, valid)
    edges = cv2.Canny(binary, 50, 150)
    h, w = gray.shape
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=int(0.15 * min(h, w)), maxLineGap=15)
    if lines is None:
        return img, 0.0
    angs, wts = [], []
    for line in lines:
        line = np.asarray(line).reshape(-1)
        if line.size != 4:
            continue
        x1, y1, x2, y2 = line.astype(float)
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        a = (a + 45) % 90 - 45            # map to [-45, 45): deviation from nearest axis
        if abs(a) <= 15:                  # ignore implausible angles
            angs.append(a)
            wts.append(np.hypot(x2 - x1, y2 - y1))
    if not angs:
        return img, 0.0
    angs, wts = np.array(angs), np.array(wts)
    order = np.argsort(angs)
    cw = np.cumsum(wts[order])
    angle = float(angs[order][np.searchsorted(cw, cw[-1] / 2)])  # weighted median
    if abs(angle) < 0.3:
        return img, 0.0
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rot = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rot, angle


# ------------------------- grid line detection -----------------------------
def line_masks(gray, valid):
    binary = bright_binary(gray, valid)
    L = max(25, int(0.10 * min(gray.shape)))
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
    vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))
    return horiz, vert


def coverage(mask, valid, axis):
    """Fraction of each row (axis=1) / column (axis=0) covered by line pixels."""
    num = (mask > 0).sum(axis=axis).astype(float)
    den = (valid > 0).sum(axis=axis).astype(float)
    cov = num / np.maximum(den, 1)
    cov[den < 0.3 * den.max()] = 0
    return cov


def find_bands(cov, thr=LINE_COVER_THR, merge_gap=3):
    on = cov > thr
    bands, i, n = [], 0, len(on)
    while i < n:
        if on[i]:
            j = i
            while j + 1 < n and on[j + 1]:
                j += 1
            bands.append([i, j])
            i = j + 1
        else:
            i += 1
    merged = []
    for b in bands:
        if merged and b[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = b[1]
        else:
            merged.append(b)
    return merged


def cluster_bands(bands, gap):
    """Merge multi-line borders (double/triple lines) into one cluster."""
    clusters = []
    for s, e in bands:
        if clusters and s - clusters[-1]["end"] <= gap:
            clusters[-1]["end"] = e
            clusters[-1]["n"] += 1
        else:
            clusters.append({"start": s, "end": e, "n": 1})
    for c in clusters:
        c["center"] = (c["start"] + c["end"]) / 2.0
    return clusters


def select_run(clusters, n_cells):
    """
    Find the large square along one axis: n_cells+1 equally spaced borders.
    Prefers the largest pitch (the big square, not the dense sub-grids) and
    multi-line borders. A run of n_cells lines is extended by one neighbouring
    cluster (e.g. a dense band that forms the square's edge).
    Returns (lo_coord, hi_coord, pitch) = inner edges of the two borders.
    """
    if len(clusters) < n_cells:
        return None
    centers = np.array([c["center"] for c in clusters])
    d = np.diff(centers)
    cands = sorted(set(np.round(d).astype(int).tolist()), reverse=True)

    def coords(i0, i1):
        return clusters[i0]["end"] + 1, clusters[i1]["start"]

    for p in cands:
        if p <= 0:
            continue
        runs, start = [], 0
        for i in range(1, len(centers) + 1):
            if i == len(centers) or abs(d[i - 1] - p) > 0.2 * p:
                runs.append((start, i - 1))
                start = i
        options = []
        for s, e in runs:
            length = e - s + 1
            if length >= n_cells + 1:
                mid = (s + e - n_cells) / 2
                for k in range(s, e - n_cells + 1):
                    sc = (clusters[k]["n"] > 1) + (clusters[k + n_cells]["n"] > 1)
                    options.append((sc, -abs(k - mid), k, k + n_cells))
            elif length == n_cells:
                if s > 0 and abs(clusters[s]["center"] - clusters[s - 1]["end"] - p) < 0.2 * p:
                    options.append((clusters[s - 1]["n"] > 1, -1, s - 1, e))
                if e < len(clusters) - 1 and abs(clusters[e + 1]["start"] - clusters[e]["center"] - p) < 0.2 * p:
                    options.append((clusters[e + 1]["n"] > 1, -1, s, e + 1))
        if options:
            _, _, i0, i1 = max(options, key=lambda o: (o[0], o[1]))
            lo, hi = coords(i0, i1)
            return lo, hi, float(p)
    return None


def locate_square(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    valid = valid_region(gray)
    horiz, vert = line_masks(gray, valid)

    gap_y, gap_x = BAND_MERGE_FRAC * h, BAND_MERGE_FRAC * w
    rows = cluster_bands(find_bands(coverage(horiz, valid, 1)), gap_y)
    cols = cluster_bands(find_bands(coverage(vert, valid, 0)), gap_x)
    ry = select_run(rows, CELLS_PER_SIDE)
    rx = select_run(cols, CELLS_PER_SIDE)

    ok = ry is not None and rx is not None
    if ok:
        y1, y2 = int(ry[0]), int(ry[1])       # inner edges of top / bottom borders
        x1, x2 = int(rx[0]), int(rx[1])       # inner edges of left / right borders
        pitch = float(np.mean([ry[2], rx[2]]))
    else:  # fallback: central crop of valid area
        ys, xs = np.where(valid > 0)
        y1, y2 = int(np.percentile(ys, 12)), int(np.percentile(ys, 88))
        x1, x2 = int(np.percentile(xs, 12)), int(np.percentile(xs, 88))
        pitch = (min(y2 - y1, x2 - x1)) / CELLS_PER_SIDE
    return (x1, y1, x2, y2), pitch, ok, (horiz, vert)


# ------------------------------ counting -----------------------------------
def count_cells(crop, line_mask, pitch):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lines = cv2.dilate(line_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))) > 0
    med = np.median(gray[~lines]) if (~lines).any() else np.median(gray)
    g = gray.copy()
    g[lines] = med                                    # suppress grid lines
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    bg = cv2.GaussianBlur(g, (0, 0), max(5, pitch * 0.2))
    diff = cv2.absdiff(g, bg)

    t, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = max(12, t)
    binary = (diff > t).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # fill the ring-shaped cells
    pad = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff = pad.copy()
    cv2.floodFill(ff, np.zeros((pad.shape[0] + 2, pad.shape[1] + 2), np.uint8), (0, 0), 255)
    filled = (pad | cv2.bitwise_not(ff))[1:-1, 1:-1]
    filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    n, lab, stats, cents = cv2.connectedComponentsWithStats(filled)
    min_a = np.pi / 4 * (0.04 * pitch) ** 2
    max_single = np.pi / 4 * (0.14 * pitch) ** 2
    areas = stats[1:, cv2.CC_STAT_AREA]
    singles = areas[(areas >= min_a) & (areas <= max_single)]
    if len(singles) == 0:
        return 0, []
    med_area = float(np.percentile(singles, 60))

    cells = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < min_a or a > 4 * med_area:             # noise or big debris / bubbles
            continue
        cx, cy = cents[i]
        if lines[int(cy), int(cx)]:
            continue
        comp = (lab == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        per = cv2.arcLength(cnts[0], True)
        circ = 4 * np.pi * a / (per * per) if per > 0 else 0
        k = max(1, int(round(a / med_area))) if a > 1.8 * med_area else 1
        if k == 1 and circ < 0.45:                    # irregular debris
            continue
        cells.append((float(cx), float(cy), float(np.sqrt(a / np.pi)), k))
    return int(sum(c[3] for c in cells)), cells


# ------------------------------- main --------------------------------------
def process(path):
    img = load_image(path)
    if img is None:
        return None
    img, angle = deskew(img)
    (x1, y1, x2, y2), pitch, ok, (horiz, vert) = locate_square(img)

    crop = img[y1:y2, x1:x2].copy()
    lm = cv2.bitwise_or(horiz, vert)[y1:y2, x1:x2]
    total, cells = count_cells(crop, lm, pitch)

    ann = crop.copy()
    for cx, cy, r, k in cells:
        col = (0, 255, 0) if k == 1 else (0, 165, 255)
        cv2.circle(ann, (int(cx), int(cy)), max(3, int(r)), col, 2)
    cv2.putText(ann, f"Count: {total}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    dbg = img.copy()
    cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 0, 255) if ok else (255, 0, 0), 4)

    stem = Path(path).stem
    cv2.imwrite(str(OUTPUT_FOLDER / f"{stem}_crop.png"), crop)
    cv2.imwrite(str(OUTPUT_FOLDER / f"{stem}_counted.png"), ann)
    cv2.imwrite(str(OUTPUT_FOLDER / f"{stem}_debug.jpg"), dbg)

    # 4x4 large square = 1 mm x 1 mm x 0.1 mm = 1e-4 mL  ->  cells/mL = count * 1e4 * dilution
    return {
        "Filename": Path(path).name,
        "Rotation_Angle": round(angle, 2),
        "Square_Detected": ok,
        "Crop_xyxy": f"{x1},{y1},{x2},{y2}",
        "Pitch_px": round(pitch, 1),
        "Cell_Count": total,
        "Cells_per_mL": total * 1e4 * DILUTION if ok else np.nan,
    }


if __name__ == "__main__":
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    if INPUT_FOLDER.is_file():
        files = [INPUT_FOLDER] if INPUT_FOLDER.suffix.lower() in EXTENSIONS else []
    elif INPUT_FOLDER.is_dir():
        files = sorted(
            p for p in INPUT_FOLDER.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )
    else:
        raise FileNotFoundError(f"Input path does not exist: {INPUT_FOLDER}")
    results = []
    for p in files:
        r = process(p)
        if r is None:
            print(f"Skipping unreadable file: {p.name}")
            continue
        results.append(r)
        print(f"{r['Filename']:28s} angle={r['Rotation_Angle']:6.2f}  "
              f"square={'OK ' if r['Square_Detected'] else 'FALLBACK'}  count={r['Cell_Count']}")
    pd.DataFrame(results).to_csv(OUTPUT_FOLDER / "summary.csv", index=False)
    print(f"\nDone. Outputs in '{OUTPUT_FOLDER}'")
