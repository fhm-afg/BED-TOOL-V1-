"""Bed colour tool.

    python bedtool.py

Pick a bed on the left, brush over the upholstery, click a colour on the right.
Save one, or save every colour of a fabric at once.

Everything is a plain file on disk -- no project, no config, no build step.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except ImportError:  # pragma: no cover
    sys.exit("This tool needs tkinter, which normally ships with Python.\n"
             "On Windows, re-run the Python installer and tick 'tcl/tk'.")

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
DISPLAY_MAX = 760          # bed is shown at up to this many px on its long side
SWATCH_THUMB = 46
SWATCH_COLS = 4

BG = "#1e1e20"
PANEL = "#2a2a2e"
FG = "#e8e8ec"
MUTED = "#9a9aa2"
ACCENT = "#4aa3ff"


# ----------------------------------------------------------------- colour maths

def srgb_to_linear(x):
    x = np.asarray(x, np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x):
    x = np.clip(np.asarray(x, np.float32), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92,
                    1.055 * x ** (1 / 2.4) - 0.055).astype(np.float32)


def luminance(lin):
    return lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def swatch_colour(path):
    """The fabric's true colour, with the swatch photo's own lighting removed.

    These swatches are close-ups of draped fabric, so the raw average is
    skewed by whichever fold caught the softbox. The median is far more robust
    than the mean here -- a bright highlight across a third of the frame moves
    a mean a long way and a median barely at all.
    """
    img = Image.open(path).convert("RGB")
    img.thumbnail((200, 200), Image.LANCZOS)
    lin = srgb_to_linear(np.asarray(img, np.float32) / 255.0)
    med = np.median(lin.reshape(-1, 3), axis=0)
    return med.astype(np.float32)


def swatch_texture(path, cache={}):
    """The swatch's texture as a tileable multiplier averaging 1.0.

    Your swatches are close-ups of draped fabric, so they carry the photographer's
    own lighting -- a broad soft highlight and real folds. Dividing by a heavily
    blurred copy of itself removes anything that varies slowly (the lighting)
    and keeps what varies quickly (the crush, the weave, the veining). What is
    left multiplies the flat colour, so the bed's own shading still does the
    lighting.

    The tile is mirrored into a 2x2 block so it repeats without a visible seam.
    """
    key = str(path)
    if key in cache:
        return cache[key]

    img = Image.open(path).convert("RGB")
    img.thumbnail((512, 512), Image.LANCZOS)
    lin = srgb_to_linear(np.asarray(img, np.float32) / 255.0)

    # A NARROW blur is the important detail, and narrower than you would guess.
    # At 3% of the swatch width the crush marks themselves survive, and blown
    # up onto a headboard they read as diagonal smudges -- the fabric looks
    # dirty rather than plush. At 1.2% only the nap survives, which is what
    # actually says "velvet" at this size.
    sigma = max(1.5, min(lin.shape[:2]) * 0.012)
    base = cv2.GaussianBlur(lin, (0, 0), sigma)
    flat = lin / np.clip(base, 1e-4, None)

    # Centre crop: the outermost fold is usually the darkest part of these shots.
    h, w = flat.shape[:2]
    c = int(min(h, w) * 0.62)
    flat = flat[(h - c) // 2:(h - c) // 2 + c, (w - c) // 2:(w - c) // 2 + c]

    flat /= np.clip(flat.reshape(-1, 3).mean(axis=0), 1e-4, None)
    # Cap the swing tightly. +/-8% is plenty of nap; wider and single threads
    # turn into blotches once magnified onto the frame.
    flat = np.clip(flat, 0.92, 1.09)

    top = np.concatenate([flat, flat[:, ::-1]], axis=1)
    tile = np.concatenate([top, top[::-1, :]], axis=0).astype(np.float32)

    cache[key] = tile
    return tile


def _fit_tile(tile, repeat_px, cache={}):
    key = (id(tile), int(repeat_px))
    if key in cache:
        return cache[key]
    n = max(8, int(repeat_px))
    interp = cv2.INTER_AREA if n < tile.shape[0] else cv2.INTER_CUBIC
    out = cv2.resize(tile, (n, n), interpolation=interp).astype(np.float32)
    if len(cache) > 64:
        cache.clear()
    cache[key] = out
    return out


def recolour(base_u8, mask, colour_lin, brightness=1.0,
             tile=None, texture=0.0, repeat_px=150):
    """Repaint the masked area in `colour_lin`, keeping the photo's shading.

    The bed's own light and shade is what makes this read as a photograph
    rather than a fill. We normalise the masked area's brightness so a typical
    lit patch sits at 1.0, then split it: anything below 1 multiplies the new
    colour (folds, shadow, the tufting), anything above is added on top
    (highlights). The additive part matters -- multiplying a highlight into a
    saturated colour just clips it to that colour and the velvet turns to
    plastic.

    `tile` adds the fabric's own texture on top of the flat colour, at strength
    `texture` (0 = flat colour, 1 = the swatch's full variation).
    """
    if mask.max() <= 0:
        return base_u8

    lin = srgb_to_linear(base_u8.astype(np.float32) / 255.0)
    lum = luminance(lin)

    inside = mask > 0.5
    if not inside.any():
        inside = mask > 0.0
    ref = float(np.percentile(lum[inside], 85))
    if ref < 0.02:
        ref = 0.02

    shade = np.clip(lum / ref, 0.0, 4.0)
    mult = np.clip(shade, 0.0, 1.0)
    spec = np.clip(shade - 1.0, 0.0, None)

    target = np.clip(colour_lin * float(brightness), 0.0, 1.0).reshape(1, 1, 3)

    if tile is not None and texture > 0.001:
        t = _fit_tile(tile, repeat_px)
        th, tw = t.shape[:2]
        h, w = lum.shape
        grain = t[np.arange(h) % th][:, np.arange(w) % tw]
        target = target * (1.0 + float(texture) * (grain - 1.0))
        target = np.clip(target, 0.0, 1.0)

    painted = target * mult[..., None] + spec[..., None]

    a = np.clip(mask, 0.0, 1.0)[..., None]
    out = lin * (1 - a) + painted * a
    return (linear_to_srgb(out) * 255.0 + 0.5).astype(np.uint8)


def load_rgb(path):
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return np.asarray(img.convert("RGB"), np.uint8)


def _bed_sort_key(p):
    """Sort 1, 2, 10 rather than 1, 10, 2."""
    head = p.stem.split(" ", 1)[0]
    return (int(head) if head.isdigit() else 10 ** 6, p.stem.lower())


def find_images(folder):
    """Every image under `folder`, skipping our own output directories.

    The tool writes _recoloured and _masks inside the beds folder, and this
    scan is recursive -- without the underscore rule the bed list would fill
    up with renders you already made.
    """
    if not folder or not Path(folder).is_dir():
        return []
    root = Path(folder)
    out = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXT:
            continue
        if p.name.startswith("._"):
            continue
        if any(part.startswith("_") for part in p.relative_to(root).parts[:-1]):
            continue
        out.append(p)
    return sorted(out, key=_bed_sort_key)


def find_swatches(folder):
    """{fabric name: [(colour name, path)]}, one group per subfolder."""
    out = {}
    root = Path(folder) if folder else None
    if not root or not root.is_dir():
        return out
    subs = [d for d in sorted(root.iterdir()) if d.is_dir()]
    if not subs:
        files = find_images(root)
        if files:
            out[root.name] = [(p.stem, p) for p in files]
        return out
    for d in subs:
        files = find_images(d)
        if files:
            out[d.name] = [(p.stem, p) for p in files]
    return out


# ------------------------------------------------------------------------- app

class BedTool:
    def __init__(self, root, beds_dir=None, swatches_dir=None, masks_dir=None):
        self.root = root
        self.root.title("Bed colour tool")
        self.root.configure(bg=BG)

        self.beds_dir = beds_dir
        self.swatches_dir = swatches_dir
        self.masks_dir = masks_dir

        self.bed_paths = []
        self.swatches = {}
        self.swatch_flat = []          # [(fabric, name, path, colour_lin)]
        self.swatch_items = []         # canvas item ids, parallel to swatch_flat

        self.base = None               # H,W,3 uint8
        self.mask = None               # H,W float32
        self.bounds = None             # H,W uint8, where the wand may spread
        self.labels = None             # H,W int32, wand regions
        self.tool = "wand"
        self.undo_stack = []
        self.scale = 1.0
        self.current = None            # index into swatch_flat
        self.brush = 18
        self.erasing = False
        self.show_mask = True
        self._last_xy = None
        self._preview_cache = None
        self._dirty_after = None

        self._build_ui()
        self._load_beds()
        self._load_swatches()

    # ---------------------------------------------------------------- UI build

    def _build_ui(self):
        left = tk.Frame(self.root, bg=PANEL, width=210)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        tk.Label(left, text="BEDS", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=10, pady=(10, 2))
        tk.Button(left, text="Choose folder…", command=self.pick_beds_dir,
                  bg="#3a3a40", fg=FG, relief="flat", font=("Segoe UI", 8)
                  ).pack(fill="x", padx=8, pady=(0, 6))

        wrap = tk.Frame(left, bg=PANEL)
        wrap.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        sb = tk.Scrollbar(wrap)
        sb.pack(side="right", fill="y")
        self.bed_list = tk.Listbox(wrap, bg="#232327", fg=FG, relief="flat",
                                   selectbackground=ACCENT, highlightthickness=0,
                                   font=("Segoe UI", 9), yscrollcommand=sb.set,
                                   activestyle="none")
        self.bed_list.pack(side="left", fill="both", expand=True)
        sb.config(command=self.bed_list.yview)
        self.bed_list.bind("<<ListboxSelect>>", self.on_pick_bed)

        right = tk.Frame(self.root, bg=PANEL, width=250)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        tk.Label(right, text="COLOURS", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=10, pady=(10, 2))
        tk.Button(right, text="Choose folder…", command=self.pick_swatch_dir,
                  bg="#3a3a40", fg=FG, relief="flat", font=("Segoe UI", 8)
                  ).pack(fill="x", padx=8, pady=(0, 6))

        pw = tk.Frame(right, bg=PANEL)
        pw.pack(fill="both", expand=True, padx=8)
        psb = tk.Scrollbar(pw)
        psb.pack(side="right", fill="y")
        self.palette = tk.Canvas(pw, bg="#232327", highlightthickness=0,
                                 yscrollcommand=psb.set)
        self.palette.pack(side="left", fill="both", expand=True)
        psb.config(command=self.palette.yview)
        self.palette.bind("<Button-1>", self.on_pick_swatch)
        self.palette.bind("<MouseWheel>",
                          lambda e: self.palette.yview_scroll(-e.delta // 120, "units"))

        sf = tk.Frame(right, bg=PANEL)
        sf.pack(fill="x", padx=8, pady=(6, 2))

        def slider(label, lo, hi, res, init):
            tk.Label(sf, text=label, bg=PANEL, fg=MUTED,
                     font=("Segoe UI", 8)).pack(anchor="w")
            s = tk.Scale(sf, from_=lo, to=hi, resolution=res, orient="horizontal",
                         bg=PANEL, fg=FG, troughcolor="#3a3a40",
                         highlightthickness=0, relief="flat", showvalue=True,
                         font=("Segoe UI", 7), command=lambda _: self.refresh())
            s.set(init)
            s.pack(fill="x")
            return s

        self.bright = slider("Brightness", 0.5, 1.6, 0.05, 1.0)
        self.texture = slider("Texture", 0.0, 2.5, 0.05, 1.0)
        self.tex_scale = slider("Texture size", 40, 500, 10, 200)

        tk.Button(right, text="Save mask", command=self.save_mask,
                  bg="#3a3a40", fg=FG, relief="flat",
                  font=("Segoe UI", 9)).pack(fill="x", padx=8, pady=(10, 3))
        tk.Button(right, text="Save this colour", command=self.save_one,
                  bg=ACCENT, fg="#0b2136", relief="flat",
                  font=("Segoe UI", 9, "bold")).pack(fill="x", padx=8, pady=(8, 3))
        tk.Button(right, text="Save whole fabric", command=self.save_fabric,
                  bg="#3a3a40", fg=FG, relief="flat",
                  font=("Segoe UI", 9)).pack(fill="x", padx=8, pady=3)
        tk.Button(right, text="Save every colour", command=self.save_all,
                  bg="#3a3a40", fg=FG, relief="flat",
                  font=("Segoe UI", 9)).pack(fill="x", padx=8, pady=(3, 10))

        bar = tk.Frame(self.root, bg=PANEL, height=44)
        bar.pack(side="bottom", fill="x")
        bar.pack_propagate(False)

        def tool_btn(text, cmd, w=9):
            b = tk.Button(bar, text=text, command=cmd, bg="#3a3a40", fg=FG,
                          relief="flat", width=w, font=("Segoe UI", 9))
            b.pack(side="left", padx=4, pady=8)
            return b

        self.wand_btn = tool_btn("Wand", self.set_wand)
        self.paint_btn = tool_btn("Brush", self.set_paint)
        self.erase_btn = tool_btn("Eraser", self.set_erase)
        tool_btn("Regions", self.show_regions)
        tool_btn("Undo", self.undo)
        tool_btn("Clear", self.clear_mask)
        self.mask_btn = tool_btn("Hide mask", self.toggle_mask, 11)

        tk.Label(bar, text="Wand detail", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 8)).pack(side="left", padx=(12, 2))
        self.wand_tol = tk.Scale(bar, from_=1, to=20, orient="horizontal",
                                 bg=PANEL, fg=FG, troughcolor="#3a3a40",
                                 highlightthickness=0, relief="flat",
                                 length=110, showvalue=True, font=("Segoe UI", 7),
                                 command=lambda _: self._invalidate_labels())
        # Higher = more edges respected = smaller, safer regions.
        self.wand_tol.set(12)
        self.wand_tol.pack(side="left", pady=6)

        tk.Label(bar, text="Size", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 8)).pack(side="left", padx=(12, 2))
        self.size_slider = tk.Scale(bar, from_=2, to=70, orient="horizontal",
                                    bg=PANEL, fg=FG, troughcolor="#3a3a40",
                                    highlightthickness=0, relief="flat",
                                    length=130, showvalue=True,
                                    font=("Segoe UI", 7),
                                    command=lambda v: setattr(self, "brush", int(v)))
        self.size_slider.set(self.brush)
        self.size_slider.pack(side="left", pady=6)

        self.status = tk.Label(bar, text="Pick a bed to start.", bg=PANEL,
                               fg=MUTED, font=("Segoe UI", 9), anchor="e")
        self.status.pack(side="right", padx=12)

        self.canvas = tk.Canvas(self.root, bg="#141416", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.on_down)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_up)
        self.canvas.bind("<Button-3>", self.on_down_erase)
        self.canvas.bind("<B3-Motion>", self.on_drag_erase)
        self.canvas.bind("<ButtonRelease-3>", self.on_up)

        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<bracketleft>", lambda e: self.nudge_brush(-4))
        self.root.bind("<bracketright>", lambda e: self.nudge_brush(4))
        self.root.bind("e", lambda e: self.set_erase())
        self.root.bind("b", lambda e: self.set_paint())
        self.root.bind("w", lambda e: self.set_wand())
        self.root.bind("<Control-s>", lambda e: self.save_mask())

        self._sync_tool_buttons()

    # ------------------------------------------------------------------ loading

    def _load_beds(self):
        self.bed_paths = find_images(self.beds_dir)
        self.bed_list.delete(0, "end")
        for p in self.bed_paths:
            self.bed_list.insert("end", "  " + p.stem)
        if self.bed_paths:
            self.set_status(f"{len(self.bed_paths)} beds found.")
        else:
            self.set_status("No beds — use 'Choose folder…' on the left.")

    def _load_swatches(self):
        self.swatches = find_swatches(self.swatches_dir)
        self.swatch_flat, self.swatch_items = [], []
        self.palette.delete("all")
        self._thumbs = []

        y = 8
        for fabric, entries in self.swatches.items():
            self.palette.create_text(8, y, text=fabric.upper(), anchor="nw",
                                     fill=MUTED, font=("Segoe UI", 8, "bold"))
            y += 18
            for i, (name, path) in enumerate(entries):
                col = i % SWATCH_COLS
                if col == 0 and i:
                    y += SWATCH_THUMB + 6
                x = 8 + col * (SWATCH_THUMB + 6)
                try:
                    im = Image.open(path).convert("RGB")
                    im.thumbnail((SWATCH_THUMB * 3, SWATCH_THUMB * 3), Image.LANCZOS)
                    im = im.resize((SWATCH_THUMB, SWATCH_THUMB), Image.LANCZOS)
                    photo = ImageTk.PhotoImage(im)
                    self._thumbs.append(photo)
                    item = self.palette.create_image(x, y, image=photo, anchor="nw")
                except Exception:
                    item = self.palette.create_rectangle(
                        x, y, x + SWATCH_THUMB, y + SWATCH_THUMB, fill="#444")
                self.palette.itemconfig(item, tags=(f"sw{len(self.swatch_flat)}",))
                self.swatch_items.append(item)
                self.swatch_flat.append([fabric, name, path, None])
            y += SWATCH_THUMB + 14

        self.palette.configure(scrollregion=self.palette.bbox("all"))
        n = len(self.swatch_flat)
        if n:
            self.set_status(f"{n} colours across {len(self.swatches)} fabrics.")
        else:
            self.set_status("No colours — use 'Choose folder…' on the right.")

    def pick_beds_dir(self):
        d = filedialog.askdirectory(title="Folder containing the bed photos")
        if d:
            self.beds_dir = d
            self._load_beds()

    def pick_swatch_dir(self):
        d = filedialog.askdirectory(title="Folder of fabric folders")
        if d:
            self.swatches_dir = d
            self._load_swatches()

    # ------------------------------------------------------------------ actions

    def on_pick_bed(self, _evt=None):
        sel = self.bed_list.curselection()
        if not sel:
            return
        path = self.bed_paths[sel[0]]
        self.base = load_rgb(path)
        self.bed_path = path
        h, w = self.base.shape[:2]
        self.scale = min(DISPLAY_MAX / max(h, w), 4.0)
        self.mask = np.zeros((h, w), np.float32)
        self.bounds = None
        self.labels = None
        self.undo_stack = []

        # The starter mask loads VISIBLY as the mask, as well as fencing the
        # wand in. An earlier version kept it as an invisible boundary only,
        # which made it look as though the masks had disappeared.
        saved = self._mask_path(path)
        src, note = None, ""
        if saved.exists():
            src, note = saved, "your saved mask"
        elif self.masks_dir:
            cand = Path(self.masks_dir) / f"{path.stem}.png"
            if cand.exists():
                src, note = cand, "starter mask (whole bed)"

        if src is not None:
            try:
                m = load_rgb(src)
                if m.shape[:2] == (h, w):
                    self.mask = (m.max(axis=2) > 128).astype(np.float32)
                    self.bounds = (self.mask > 0.5).astype(np.uint8)
            except Exception:
                pass

        if self.mask.max() > 0:
            self.set_status(f"{path.stem} — {note} loaded "
                            f"({self.mask.mean() * 100:.0f}% of the image). "
                            "Right-drag with the Wand to take the bedding out.")
        else:
            self.set_status(f"{path.stem} — no mask. Brush or wand the "
                            "upholstery.")
        self.refresh()

    def _mask_path(self, bed_path):
        d = Path(self.beds_dir or ".") / "_masks"
        return d / f"{bed_path.stem}.png"

    def save_mask(self):
        if self.base is None or self.mask is None:
            return
        if self.mask.max() <= 0:
            messagebox.showinfo("Nothing to save", "Mask is empty.")
            return
        p = self._mask_path(self.bed_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        out = np.zeros(self.base.shape, np.uint8)
        out[self.mask > 0.5] = (255, 0, 0)
        Image.fromarray(out).save(p)
        self.set_status(f"Mask saved — reloads automatically next time.")

    def on_pick_swatch(self, evt):
        x = self.palette.canvasx(evt.x)
        y = self.palette.canvasy(evt.y)
        hit = self.palette.find_closest(x, y)
        if not hit:
            return
        item = hit[0]
        if item not in self.swatch_items:
            return
        idx = self.swatch_items.index(item)
        # Ignore clicks that landed in the gap rather than on a thumbnail.
        bbox = self.palette.bbox(item)
        if not (bbox and bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]):
            return

        entry = self.swatch_flat[idx]
        if entry[3] is None:
            entry[3] = swatch_colour(entry[2])
        self.current = idx

        for i, it in enumerate(self.swatch_items):
            self.palette.itemconfig(it, state="normal")
        self.palette.delete("sel")
        b = self.palette.bbox(item)
        if b:
            self.palette.create_rectangle(b[0] - 2, b[1] - 2, b[2] + 2, b[3] + 2,
                                          outline=ACCENT, width=2, tags="sel")
        self.set_status(f"{entry[0]} / {entry[1]}")
        self.refresh()

    # -------------------------------------------------------------------- brush

    def _invalidate_labels(self):
        self.labels = None

    def set_wand(self):
        self.tool, self.erasing = "wand", False
        self._sync_tool_buttons()

    def set_paint(self):
        self.tool, self.erasing = "brush", False
        self._sync_tool_buttons()

    def set_erase(self):
        self.tool, self.erasing = "brush", True
        self._sync_tool_buttons()

    def _sync_tool_buttons(self):
        on = (ACCENT, "#0b2136")
        off = ("#3a3a40", FG)
        w = on if self.tool == "wand" else off
        b = on if (self.tool == "brush" and not self.erasing) else off
        e = on if (self.tool == "brush" and self.erasing) else off
        self.wand_btn.config(bg=w[0], fg=w[1])
        self.paint_btn.config(bg=b[0], fg=b[1])
        self.erase_btn.config(bg=e[0], fg=e[1])

    def _build_labels(self):
        """Carve the bed into regions separated by edges.

        The first version of this used a colour-tolerance flood fill, and it
        did not work. That kind of fill asks "is this pixel the same colour as
        where I started?" -- and on a cream bed the answer stays yes all the
        way across the seam and into the duvet. There was a cliff at tolerance
        8: below it a click got one channel, above it the entire bed.

        Edges are the right signal. The seam between frame and bedding is a
        real gradient even when both sides are identical cream, so cutting on
        gradient stops where colour never would. It also fails safely: too
        sensitive just means smaller regions and another click, rather than
        the whole bed turning pink.
        """
        if self.base is None:
            return
        # Regions cover the WHOLE image, not just the starter mask. Several
        # starter masks miss the headboard, and if the wand were fenced inside
        # them you could never click it back. Edges stop a click at the
        # headboard's own outline anyway, so there is nothing to protect
        # against -- and Undo covers a misclick.
        inside = np.ones(self.base.shape[:2], bool)

        gray = cv2.cvtColor(self.base, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray = cv2.bilateralFilter(gray, 5, 30, 5)      # denoise, keep edges
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.hypot(gx, gy)

        pct = 88.0 - 1.9 * float(self.wand_tol.get())
        thr = np.percentile(grad[inside], np.clip(pct, 35, 95))
        free = ((grad <= thr) & inside).astype(np.uint8)
        free = cv2.morphologyEx(free, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        n, labels = cv2.connectedComponents(free, 8)

        # Hand the edge pixels themselves to the nearest region, so clicking a
        # seam still does something and regions meet cleanly.
        gap = (labels == 0) & inside
        if gap.any() and n > 1:
            _, near = cv2.distanceTransformWithLabels(
                (labels == 0).astype(np.uint8), cv2.DIST_L2, 3,
                labelType=cv2.DIST_LABEL_PIXEL)
            ys, xs = np.nonzero(labels > 0)
            if len(ys):
                lut = np.zeros(int(near.max()) + 1, np.int32)
                lut[near[ys, xs]] = labels[ys, xs]
                labels = np.where(gap, lut[near], labels)
        labels[~inside] = 0
        self.labels = labels

    def _ensure_labels(self):
        if self.labels is None:
            self._build_labels()
        return self.labels

    def wand(self, x, y, remove=False):
        """Add or remove the whole region under the cursor in one go."""
        if self.base is None or self.mask is None:
            return
        h, w = self.mask.shape
        if not (0 <= x < w and 0 <= y < h):
            return

        labels = self._ensure_labels()
        if labels is None:
            return
        lab = int(labels[y, x])
        if lab == 0:
            self.set_status("Nothing there — outside the bed, or use the Brush.")
            return

        region = labels == lab
        self._push_undo()
        self.mask[region] = 0.0 if remove else 1.0
        if self.bounds is not None and self.bounds.any():
            covered = self.mask[self.bounds > 0].mean() * 100
            self.set_status(f"Mask covers {covered:.0f}% of the bed")
        self.refresh()

    def nudge_brush(self, d):
        self.brush = int(np.clip(self.brush + d, 2, 70))
        self.size_slider.set(self.brush)

    def toggle_mask(self):
        self.show_mask = not self.show_mask
        self.mask_btn.config(text="Show mask" if not self.show_mask else "Hide mask")
        self.refresh()

    def _push_undo(self):
        if self.mask is None:
            return
        self.undo_stack.append(self.mask.copy())
        if len(self.undo_stack) > 24:
            self.undo_stack.pop(0)

    def undo(self):
        if self.undo_stack:
            self.mask = self.undo_stack.pop()
            self.refresh()

    def clear_mask(self):
        if self.mask is None:
            return
        self._push_undo()
        self.mask[:] = 0.0
        self.refresh()

    def fill_mask(self):
        if self.mask is None:
            return
        self._push_undo()
        self.mask[:] = 1.0
        self.refresh()

    def _img_xy(self, evt):
        return int(self.canvas.canvasx(evt.x) / self.scale), \
               int(self.canvas.canvasy(evt.y) / self.scale)

    def on_down(self, evt):
        if self.mask is None:
            return
        x, y = self._img_xy(evt)
        if self.tool == "wand":
            self.wand(x, y, remove=False)
            return
        self._push_undo()
        self._last_xy = (x, y)
        self._stroke(self._last_xy, self._last_xy, self.erasing)

    def on_down_erase(self, evt):
        if self.mask is None:
            return
        x, y = self._img_xy(evt)
        if self.tool == "wand":
            self.wand(x, y, remove=True)
            return
        self._push_undo()
        self._last_xy = (x, y)
        self._stroke(self._last_xy, self._last_xy, True)

    def on_drag(self, evt):
        if self.tool == "wand":
            self._wand_drag(evt, False)
            return
        self._continue(evt, self.erasing)

    def on_drag_erase(self, evt):
        if self.tool == "wand":
            self._wand_drag(evt, True)
            return
        self._continue(evt, True)

    def _wand_drag(self, evt, remove):
        """Wand on drag: sweep across a tufted headboard and every channel
        fills. Each channel is walled off from its neighbours by a shadow line,
        so one click only ever gets one channel -- dragging is what makes the
        wand practical on this furniture."""
        x, y = self._img_xy(evt)
        h, w = self.mask.shape
        if not (0 <= x < w and 0 <= y < h):
            return
        # Skip points already in the state we want, so a slow drag is cheap.
        if (self.mask[y, x] > 0.5) != remove:
            return
        self.wand(x, y, remove=remove)

    def show_regions(self):
        """Flash the wand's regions, so you can see what a click will grab."""
        labels = self._ensure_labels()
        if labels is None or self.base is None:
            return
        rng = np.random.default_rng(4)
        pal = rng.integers(60, 255, (int(labels.max()) + 2, 3)).astype(np.uint8)
        pal[0] = 0
        col = pal[np.clip(labels, 0, len(pal) - 1)]
        vis = np.where((labels > 0)[..., None],
                       (self.base * 0.35 + col * 0.65).astype(np.uint8), self.base)
        pil = Image.fromarray(vis)
        w = max(1, int(pil.width * self.scale))
        h = max(1, int(pil.height * self.scale))
        self._photo = ImageTk.PhotoImage(pil.resize((w, h), Image.LANCZOS))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self._photo, anchor="nw")
        n = len(np.unique(labels)) - 1
        self.set_status(f"{n} regions — each colour is one wand click. "
                        "Click anywhere to go back.")
        self.canvas.bind("<Motion>", self._end_regions)

    def _end_regions(self, _evt=None):
        self.canvas.unbind("<Motion>")
        self.refresh()

    def _continue(self, evt, erase):
        if self.mask is None or self._last_xy is None:
            return
        xy = self._img_xy(evt)
        self._stroke(self._last_xy, xy, erase)
        self._last_xy = xy

    def on_up(self, _evt):
        self._last_xy = None

    def _stroke(self, a, b, erase):
        """Draw a thick line into the mask, so fast drags stay continuous."""
        r = max(1, int(self.brush / 2))
        value = 0.0 if erase else 1.0
        cv2.line(self.mask, a, b, value, thickness=r * 2, lineType=cv2.LINE_8)
        cv2.circle(self.mask, b, r, value, -1, lineType=cv2.LINE_8)
        self._schedule_refresh()

    def _schedule_refresh(self):
        """Coalesce redraws while the mouse is moving, so painting stays smooth."""
        if self._dirty_after is not None:
            return
        self._dirty_after = self.root.after(16, self._do_scheduled)

    def _do_scheduled(self):
        self._dirty_after = None
        self.refresh()

    # ------------------------------------------------------------------ display

    def composed(self):
        if self.base is None:
            return None
        if self.current is None or self.mask is None or self.mask.max() <= 0:
            return self.base
        _, _, path, colour = self.swatch_flat[self.current]
        tex = float(self.texture.get())
        return recolour(self.base, self.mask, colour, float(self.bright.get()),
                        tile=swatch_texture(path) if tex > 0.001 else None,
                        texture=tex, repeat_px=int(self.tex_scale.get()))

    def refresh(self):
        img = self.composed()
        if img is None:
            return
        show = img
        if self.show_mask and self.mask is not None and self.mask.max() > 0 \
                and self.current is None:
            a = (self.mask * 0.45)[..., None]
            show = np.clip(img * (1 - a) + np.array([255, 0, 160], np.float32) * a,
                           0, 255).astype(np.uint8)

        pil = Image.fromarray(show)
        w = max(1, int(pil.width * self.scale))
        h = max(1, int(pil.height * self.scale))
        pil = pil.resize((w, h), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self._photo, anchor="nw")
        self.canvas.configure(scrollregion=(0, 0, w, h))

    def set_status(self, text):
        self.status.config(text=text)
        self.root.update_idletasks()

    # -------------------------------------------------------------------- saving

    def _out_dir(self):
        base = Path(self.beds_dir or ".") / "_recoloured"
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _guard(self):
        if self.base is None:
            messagebox.showinfo("Nothing to save", "Pick a bed first.")
            return False
        if self.mask is None or self.mask.max() <= 0:
            messagebox.showinfo("Nothing masked",
                                "Brush over the upholstery first.")
            return False
        return True

    def save_one(self):
        if not self._guard():
            return
        if self.current is None:
            messagebox.showinfo("No colour", "Click a colour on the right.")
            return
        fabric, name, path, colour = self.swatch_flat[self.current]
        out = self._out_dir() / f"{self.bed_path.stem}__{fabric}__{name}.png"
        tex = float(self.texture.get())
        Image.fromarray(recolour(
            self.base, self.mask, colour, float(self.bright.get()),
            tile=swatch_texture(path) if tex > 0.001 else None,
            texture=tex, repeat_px=int(self.tex_scale.get()))).save(out)
        self.set_status(f"Saved {out.name}")

    def _save_many(self, entries, label):
        if not self._guard():
            return
        out_dir = self._out_dir()
        bright = float(self.bright.get())
        tex = float(self.texture.get())
        scale = int(self.tex_scale.get())
        t0 = time.perf_counter()
        for i, entry in enumerate(entries, 1):
            fabric, name, path, colour = entry
            if colour is None:
                colour = swatch_colour(path)
                entry[3] = colour
            img = recolour(self.base, self.mask, colour, bright,
                           tile=swatch_texture(path) if tex > 0.001 else None,
                           texture=tex, repeat_px=scale)
            Image.fromarray(img).save(
                out_dir / f"{self.bed_path.stem}__{fabric}__{name}.png")
            if i % 3 == 0 or i == len(entries):
                self.set_status(f"Saving {label}… {i}/{len(entries)}")
        dt = time.perf_counter() - t0
        self.set_status(f"Saved {len(entries)} images to _recoloured/ in {dt:.1f}s")

    def save_fabric(self):
        if self.current is None:
            messagebox.showinfo("No colour", "Click any colour of the fabric first.")
            return
        fabric = self.swatch_flat[self.current][0]
        self._save_many([e for e in self.swatch_flat if e[0] == fabric], fabric)

    def save_all(self):
        self._save_many(list(self.swatch_flat), "everything")


def load_mask(path, shape):
    """Read a mask PNG: red (or any non-black) marks the area to recolour."""
    m = load_rgb(path)
    if m.shape[:2] != shape:
        return None
    red = (m[..., 0].astype(np.int16) - np.maximum(m[..., 1], m[..., 2])) > 60
    if red.any():
        return red.astype(np.float32)
    grey = m.max(axis=2) > 128
    return grey.astype(np.float32) if grey.any() else None


def batch(beds_dir, swatches_dir, masks_dir, out_dir,
          only_bed=None, only_fabric=None, brightness=1.0, dry_run=False,
          texture=0.85, repeat_px=150):
    """Headless render: every masked bed against every colour.

    Same maths as the button in the window, so what you approve on screen is
    exactly what comes out of a batch.
    """
    beds = find_images(beds_dir)
    groups = find_swatches(swatches_dir)
    if only_bed:
        beds = [p for p in beds if only_bed.lower() in p.stem.lower()]
    if only_fabric:
        groups = {k: v for k, v in groups.items() if only_fabric.lower() in k.lower()}

    flat = [(fab, name, path) for fab, entries in groups.items()
            for name, path in entries]
    if not beds or not flat:
        print("nothing to do -- check --beds and --swatches", file=sys.stderr)
        return 1

    masks_dir = Path(masks_dir) if masks_dir else None
    out_dir = Path(out_dir)

    # Masks you saved in the window beat the rough auto ones.
    saved_dir = Path(beds_dir) / "_masks"

    usable, skipped = [], []
    for bed in beds:
        mp = saved_dir / f"{bed.stem}.png"
        if not mp.exists():
            if not masks_dir:
                skipped.append((bed.stem, "no saved mask and no --masks folder"))
                continue
            mp = masks_dir / f"{bed.stem}.png"
        if not mp.exists():
            skipped.append((bed.stem, "no mask file"))
            continue
        base = load_rgb(bed)
        mask = load_mask(mp, base.shape[:2])
        if mask is None:
            skipped.append((bed.stem, "mask empty or wrong size"))
            continue
        usable.append((bed, base, mask))

    total = len(usable) * len(flat)
    print(f"{len(usable)} masked bed(s) x {len(flat)} colour(s) = {total} images")
    for stem, why in skipped:
        print(f"  skipped {stem}: {why}")
    if dry_run or not usable:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    colours = {}
    done = 0
    t0 = time.perf_counter()
    for bed, base, mask in usable:
        for fab, name, path in flat:
            if path not in colours:
                colours[path] = swatch_colour(path)
            img = recolour(base, mask, colours[path], brightness,
                           tile=swatch_texture(path) if texture > 0.001 else None,
                           texture=texture, repeat_px=repeat_px)
            Image.fromarray(img).save(out_dir / f"{bed.stem}__{fab}__{name}.png")
            done += 1
        print(f"  {bed.stem}: {len(flat)} colours")
    dt = time.perf_counter() - t0
    print(f"\n{done} images in {dt:.1f}s ({dt / max(1, done):.2f}s each) -> {out_dir}/")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--beds", help="folder of bed photos")
    ap.add_argument("--swatches", help="folder containing one subfolder per fabric")
    ap.add_argument("--masks", help="folder of mask PNGs, named to match the beds")
    ap.add_argument("--batch", action="store_true",
                    help="render everything without opening the window")
    ap.add_argument("--out", default="_recoloured", help="output folder for --batch")
    ap.add_argument("--bed", help="--batch: only beds whose name contains this")
    ap.add_argument("--fabric", help="--batch: only fabrics whose name contains this")
    ap.add_argument("--brightness", type=float, default=1.0)
    ap.add_argument("--texture", type=float, default=0.85,
                    help="fabric texture strength, 0 = flat colour (default 0.85)")
    ap.add_argument("--texture-size", type=int, default=150,
                    help="pixels per texture repeat (default 150)")
    ap.add_argument("--dry-run", action="store_true", help="--batch: count only")
    args = ap.parse_args(argv)

    if args.batch:
        return batch(args.beds, args.swatches, args.masks, args.out,
                     args.bed, args.fabric, args.brightness, args.dry_run,
                     args.texture, args.texture_size)

    root = tk.Tk()
    root.geometry("1360x860")
    BedTool(root, args.beds, args.swatches, args.masks)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
