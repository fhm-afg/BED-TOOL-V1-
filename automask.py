"""Generate starter region maps: the bed separated from the room, automatically.

    python automask.py --in <bed images dir> --out <output dir>
    python automask.py --in beds --out masks --only "1 OAKLEY"

For each bed image this writes:

    regions/<name>.png   region map, whole bed filled #ff0000
    review/<name>.png     the mask painted over the photo, for checking
    review/_contact.jpg   every bed at a glance
    report.txt            per-bed stats, suspicious ones flagged

The mask covers the WHOLE bed -- frame and bedding together -- because cream
upholstery and white bedding measure the same brightness and cannot be told
apart automatically. Finishing the job means painting the duvet and pillows
back to BLACK in the region map, leaving red on the upholstery only. That is
the one step a person has to do, and it is minutes per bed rather than hours,
because the outline is already correct.

Where the upholstery is dark and the bedding pale, --split attempts the
frame/bedding separation too; the report says how confident it is.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
FRAME_KEY = (255, 0, 0)
BEDDING_KEY = (0, 255, 0)


def load_rgb(path):
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def fill_holes(mask):
    """Close interior gaps -- pillows read as background otherwise."""
    inv = (1 - mask).astype(np.uint8)
    n, labels, _, _ = cv2.connectedComponentsWithStats(inv, 8)
    if n <= 1:
        return mask
    border = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
    out = mask.copy()
    for i in range(1, n):
        if i not in border:
            out[labels == i] = 1
    return out


def segment_bed(bgr, top=0.16, bottom=0.99, side=0.02, iters=9, rect=None,
                refine=False):
    """Segment the bed. `rect` is a fractional (x, y, w, h) box around the bed.

    The default full-width box only works when the bed dominates the frame. In
    a wide room shot grabcut latches onto whatever fills the box instead --
    curtains, a rug, a feature wall -- so a per-bed box is what makes this
    reliable.
    """
    h, w = bgr.shape[:2]
    if rect is not None:
        fx, fy, fw, fh = rect
        x0 = max(1, int(w * fx))
        y0 = max(1, int(h * fy))
        rw = max(4, min(int(w * fw), w - x0 - 1))
        rh = max(4, min(int(h * fh), h - y0 - 1))
    else:
        x0 = max(1, int(w * side))
        y0 = max(1, int(h * top))
        rw = max(4, int(w * (1 - 2 * side)))
        rh = max(4, int(h * bottom) - y0)

    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, (x0, y0, rw, rh), bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    m = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)

    # Second pass. The first one models foreground purely by colour, so a black
    # headboard gets read as more of the dark room and dropped -- which is
    # exactly what happened to the Valleta, Washington and Sovereign beds.
    # Re-running with the first result pinned as definite foreground lets the
    # dark parts join the bed they are attached to.
    if refine and m.sum() > 200:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        sure = cv2.erode(m, k, 1)
        seed = np.full((h, w), cv2.GC_BGD, np.uint8)
        seed[y0:y0 + rh, x0:x0 + rw] = cv2.GC_PR_FGD
        seed[m > 0] = cv2.GC_PR_FGD
        seed[sure > 0] = cv2.GC_FGD
        try:
            bgd2, fgd2 = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            cv2.grabCut(bgr, seed, None, bgd2, fgd2, 5, cv2.GC_INIT_WITH_MASK)
            m2 = np.where((seed == cv2.GC_FGD) | (seed == cv2.GC_PR_FGD),
                          1, 0).astype(np.uint8)
            # Keep it only if it grew sensibly rather than swallowing the room.
            if m.sum() < m2.sum() <= m.sum() * 3.2:
                m = m2
        except cv2.error:
            pass

    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        m = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    m = fill_holes(m)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return m


def _flood_from(bgr_blur, bed, seed, tol):
    """One flood fill, walled in by the bed silhouette."""
    h, w = bed.shape
    ff = np.ones((h + 2, w + 2), np.uint8)
    ff[1:-1, 1:-1] = (bed == 0).astype(np.uint8)
    cv2.floodFill(bgr_blur.copy(), ff, seed, 255, (tol,) * 3, (tol,) * 3,
                  4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
    return ((ff[1:-1, 1:-1] == 255) & (bed > 0)).astype(np.uint8)


def _compactness(m):
    """Area over squared perimeter. A duvet is a blob; a leak is stringy."""
    area = float(m.sum())
    if area < 20:
        return 0.0
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    per = sum(cv2.arcLength(c, True) for c in cnts)
    return 0.0 if per < 1 else float(4 * np.pi * area / (per * per))


def split_by_flood(bgr, bed, lo=0.22, hi=0.68):
    """Separate bedding from upholstery with flood fills, seeds chosen by search.

    Brightness cannot tell cream upholstery from white bedding, but the two are
    nearly always divided by a seam or a contact shadow -- and a flood fill only
    needs a boundary, not a colour difference.

    We do not know where the bedding is, so we try a grid of seeds at several
    tolerances and keep whichever fill looks most like bedding: a compact blob
    covering a plausible share of the bed. A leak through a gap into the frame
    is stringy, so compactness rejects it.
    """
    bed = (bed > 0).astype(np.uint8)
    if bed.sum() < 400:
        return None, None, 0.0

    blur = cv2.GaussianBlur(bgr, (0, 0), 1.0)
    ys, xs = np.nonzero(bed)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()

    seeds = []
    for fy in np.linspace(0.15, 0.8, 7):
        for fx in np.linspace(0.15, 0.85, 7):
            y = int(y0 + (y1 - y0) * fy)
            x = int(x0 + (x1 - x0) * fx)
            if bed[y, x]:
                seeds.append((x, y))
    if not seeds:
        return None, None, 0.0

    best = None
    for tol in (5, 8, 11, 15):
        for seed in seeds:
            fill = _flood_from(blur, bed, seed, tol)
            share = fill.sum() / bed.sum()
            if not (lo <= share <= hi):
                continue
            score = _compactness(fill)
            if best is None or score > best[0]:
                best = (score, fill, share, tol, seed)
    if best is None:
        return None, None, 0.0

    score, bedding, share, tol, _ = best

    # Pillows often sit apart from the duvet, so absorb any further compact
    # fill that does not push the total past the plausible ceiling.
    for seed in seeds:
        if bedding[seed[1], seed[0]]:
            continue
        extra = _flood_from(blur, bed, seed, tol)
        es = extra.sum() / bed.sum()
        if es < 0.02 or (bedding | extra).sum() / bed.sum() > hi:
            continue
        if _compactness(extra) > 0.05:
            bedding |= extra

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bedding = cv2.morphologyEx(bedding, cv2.MORPH_CLOSE, k)
    bedding &= bed
    uphol = ((bed > 0) & (bedding == 0)).astype(np.uint8)
    uphol = cv2.morphologyEx(uphol, cv2.MORPH_OPEN, k)
    return uphol, bedding, score


def try_split(bgr, bed, min_gap=22.0):
    """Split bed into (upholstery, bedding) when brightness actually separates.

    Returns (uphol, bedding, gap). A gap under min_gap L* units means the two
    are the same tone and the split is not trustworthy -- caller should ignore
    it rather than ship a wrong guess.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.float32)
    inside = bed > 0
    if inside.sum() < 200:
        return None, None, 0.0

    samples = L[inside].reshape(-1, 1).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
    _, labels, centres = cv2.kmeans(samples, 2, None, crit, 8, cv2.KMEANS_PP_CENTERS)
    centres = centres.ravel()
    gap = float(abs(centres[0] - centres[1]))

    dark_is = int(np.argmin(centres))
    full = np.full(bed.shape, -1, np.int32)
    full[inside] = labels.ravel()

    uphol = (full == dark_is).astype(np.uint8)
    bedding = (full == (1 - dark_is)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    uphol = cv2.morphologyEx(uphol, cv2.MORPH_OPEN, k)
    bedding = cv2.morphologyEx(bedding, cv2.MORPH_OPEN, k)
    return uphol, bedding, gap


def overlay(rgb, mask, tint, alpha=0.45):
    out = rgb.astype(np.float32)
    a = (mask.astype(np.float32) * alpha)[..., None]
    return np.clip(out * (1 - a) + np.array(tint, np.float32) * a, 0, 255).astype(np.uint8)


def contact_sheet(entries, cols=7, thumb_w=260):
    if not entries:
        return None
    rows = (len(entries) + cols - 1) // cols
    label_h, gap = 16, 5
    probe = max(entries, key=lambda e: e[1].shape[0])[1]
    thumb_h = max(1, round(thumb_w * probe.shape[0] / probe.shape[1]))
    cell_h = thumb_h + label_h

    sheet = Image.new("RGB", (cols * thumb_w + (cols + 1) * gap,
                              rows * cell_h + (rows + 1) * gap), (26, 26, 28))
    draw = ImageDraw.Draw(sheet)
    for i, (name, img) in enumerate(entries):
        r, c = divmod(i, cols)
        x, y = gap + c * (thumb_w + gap), gap + r * (cell_h + gap)
        thumb = Image.fromarray(img).resize((thumb_w, thumb_h), Image.LANCZOS)
        sheet.paste(thumb, (x, y))
        draw.text((x + 2, y + thumb_h + 3), name[:38], fill=(226, 226, 232))
    return sheet


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--out", dest="outdir", required=True)
    ap.add_argument("--only", help="substring filter on filename")
    ap.add_argument("--split", action="store_true",
                    help="also attempt the frame/bedding split where tones allow")
    ap.add_argument("--split-gap", type=float, default=22.0,
                    help="unused by the flood method, kept for the old one")
    ap.add_argument("--refine", action="store_true",
                    help="second grabcut pass to pull in dark headboards. "
                         "Helps occasionally, and can grab the rug -- check "
                         "the review sheet before trusting it.")
    ap.add_argument("--min-compact", type=float, default=0.06,
                    help="reject a split whose bedding blob is stringier than "
                         "this (default 0.06)")
    ap.add_argument("--top", type=float, default=0.16,
                    help="fraction of height excluded at the top (default 0.16)")
    ap.add_argument("--rects", help="JSON file: {\"<bed stem>\": [x, y, w, h], ...} "
                                    "as fractions of the image, boxing the bed")
    args = ap.parse_args(argv)

    rects = {}
    if args.rects:
        import json
        rects = json.loads(Path(args.rects).read_text(encoding="utf-8"))
        rects = {k: [float(v) for v in vals] for k, vals in rects.items()
                 if not k.startswith("_")}

    indir, outdir = Path(args.indir), Path(args.outdir)
    files = sorted(p for p in indir.rglob("*")
                   if p.is_file() and p.suffix.lower() in IMAGE_EXT
                   and not p.name.startswith("._"))
    if args.only:
        files = [p for p in files if args.only.lower() in p.name.lower()]
    if not files:
        print(f"no images found under {indir}", file=sys.stderr)
        return 1

    (outdir / "regions").mkdir(parents=True, exist_ok=True)
    (outdir / "review").mkdir(parents=True, exist_ok=True)

    rows, thumbs = [], []
    for i, path in enumerate(files, 1):
        rgb = load_rgb(path)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]

        rect = rects.get(path.stem)
        bed = segment_bed(bgr, top=args.top, rect=rect, refine=args.refine)
        share = float(bed.mean())

        rmap = np.zeros((h, w, 3), np.uint8)
        split_note = ""
        bedding = None
        if args.split:
            uphol, bedding, score = split_by_flood(bgr, bed)
            if uphol is not None and score >= args.min_compact:
                rmap[uphol > 0] = FRAME_KEY
                split_note = (f"split ok (blob {score:.2f}, "
                              f"bedding {bedding.sum() / max(1, bed.sum()) * 100:.0f}%)")
            else:
                rmap[bed > 0] = FRAME_KEY
                bedding = None
                split_note = (f"split FAILED (best blob {score:.2f}) "
                              "-- whole bed marked")
        else:
            rmap[bed > 0] = FRAME_KEY

        flags = []
        if share < 0.08:
            flags.append("TOO SMALL")
        if share > 0.75:
            flags.append("TOO LARGE")
        if bed[:2, :].any():
            flags.append("touches top")

        stem = path.stem
        Image.fromarray(rmap).save(outdir / "regions" / f"{stem}.png")

        if bedding is not None:
            vis = overlay(rgb, (rmap[..., 0] > 128).astype(np.uint8), (255, 0, 160), 0.5)
            vis = overlay(vis, bedding, (60, 220, 255), 0.4)
        else:
            vis = overlay(rgb, bed, (255, 0, 160), 0.45)
        Image.fromarray(vis).save(outdir / "review" / f"{stem}.png")
        thumbs.append((stem, vis))

        status = ", ".join(flags) if flags else "ok"
        boxed = "boxed" if rect else "auto "
        rows.append((stem, w, h, share, status, split_note))
        print(f"[{i:>2}/{len(files)}] {boxed} {stem:<36} {w:>4}x{h:<4} "
              f"bed {share * 100:>5.1f}%  {status}"
              + (f"  {split_note}" if split_note else ""))

    sheet = contact_sheet(thumbs)
    if sheet is not None:
        sheet.save(outdir / "review" / "_contact.jpg", quality=90, optimize=True)

    lines = ["bed                                    size        bed%   status",
             "-" * 78]
    for stem, w, h, share, status, note in rows:
        lines.append(f"{stem:<38} {w:>4}x{h:<4} {share * 100:>6.1f}%  {status}"
                     + (f"  |  {note}" if note else ""))
    bad = [r for r in rows if r[4] != "ok"]
    lines += ["", f"{len(rows)} beds, {len(bad)} flagged for a look"]
    (outdir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n{len(rows)} masks written to {outdir}/regions/")
    print(f"review sheet: {outdir}/review/_contact.jpg")
    if bad:
        print(f"{len(bad)} flagged: " + ", ".join(r[0] for r in bad))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
