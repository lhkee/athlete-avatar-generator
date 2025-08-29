# app.py — Athlete Image Generator (streamlit-cropper edition, corrected)
# - Real-time, reliable drag-to-pan/scale using a crop box
# - Per-size guides & red frame
# - Auto-straighten (eyes) toggle
# - TIFF (incl. LZW) supported via Pillow
# - ZIP export with naming: <base>-avatar_<WxH>.png / <base>-hero_<WxH>.png

import io
import os
import zipfile
import requests
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
import cv2
from streamlit_cropper import st_cropper

# ───────────────────────────────────────────────────────────────
# Config & guide overlays (GitHub raw URLs you provided)
GUIDE_URLS = {
    "256x256":   "https://raw.githubusercontent.com/lhkee/athlete-avatar-generator/ae2815490a0ad2861801054277022572b1a06eee/256x256-guide.png",
    "500x345":   "https://raw.githubusercontent.com/lhkee/athlete-avatar-generator/3936a686c65728e1811e9370fadd991125d91e61/500x345-guide.png",
    "1200x1165": "https://raw.githubusercontent.com/lhkee/athlete-avatar-generator/934f591fa238f6f597816307a6fc655792e15279/1200x1165-guide.png",
    "1500x920":  "https://raw.githubusercontent.com/lhkee/athlete-avatar-generator/934f591fa238f6f597816307a6fc655792e15279/1500x920-guide.png",
}

st.set_page_config(page_title="Athlete Image Generator (Cropper)", layout="wide")
st.markdown(
    """
    <style>
    div.block-container{padding-top:1rem;padding-bottom:2rem;}
    .img-frame{outline:1px solid #ff2b2b; box-shadow: inset 0 0 0 3px rgba(16,185,129,.9); width: fit-content;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("🏋️ Athlete Image Generator — Real-time crop box (pan + scale) + guides")

# ───────────────────────────────────────────────────────────────
# Cached helpers

@st.cache_data(show_spinner=False)
def load_overlay(url: str, w: int, h: int):
    if not url:
        return None
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        g = Image.open(io.BytesIO(r.content)).convert("RGBA")
        if g.size != (w, h):
            g = g.resize((w, h), Image.BICUBIC)
        return g
    except Exception:
        return None

def draw_frame(img: Image.Image) -> Image.Image:
    """1px outer red + 2px inner mint."""
    im = img.convert("RGBA")
    d = ImageDraw.Draw(im)
    w, h = im.size
    d.rectangle([(0, 0), (w - 1, h - 1)], outline=(255, 0, 0, 255), width=1)
    if w > 6 and h > 6:
        d.rectangle([(3, 3), (w - 4, h - 4)], outline=(0, 255, 160, 255), width=2)
    return im

@st.cache_data(show_spinner=False)
def load_img_from_bytes(b: bytes, max_dim: int = 4000) -> Image.Image | None:
    try:
        img = Image.open(io.BytesIO(b))
        img.load()  # ensures TIFF/LZW decoded
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        w, h = img.size
        if max(w, h) > max_dim:
            s = max_dim / float(max(w, h))
            img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
        return img
    except Exception:
        return None

# ───────────────────────────────────────────────────────────────
# Auto-straighten (eye line)
FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EYES = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def auto_straighten(pil_img: Image.Image) -> Image.Image:
    try:
        rgb = pil_img.convert("RGB")
        np_im = np.array(rgb)
        gray = cv2.cvtColor(np_im, cv2.COLOR_RGB2GRAY)

        faces = FACE.detectMultiScale(gray, 1.1, 4, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(80, 80))
        if len(faces) == 0:
            return pil_img
        x, y, w, h = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[0]
        roi = gray[y : y + h, x : x + w]
        eyes = EYES.detectMultiScale(roi, 1.1, 6, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(25, 25))
        if len(eyes) < 2:
            return pil_img
        eyes = sorted(eyes, key=lambda e: e[0])[:2]
        (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2) = eyes
        p1 = (x + ex1 + ew1 // 2, y + ey1 + eh1 // 2)
        p2 = (x + ex2 + ew2 // 2, y + ey2 + eh2 // 2)
        dy, dx = (p2[1] - p1[1]), (p2[0] - p1[0])
        angle = np.degrees(np.arctan2(dy, dx))

        h0, w0 = np_im.shape[:2]
        M = cv2.getRotationMatrix2D((w0 / 2, h0 / 2), angle, 1.0)
        rot = cv2.warpAffine(np_im, M, (w0, h0), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(rot).convert("RGBA")
    except Exception:
        return pil_img

# ───────────────────────────────────────────────────────────────
# Initial crop guess (LTRB in original coords)

def detect_face_box(pil_img: Image.Image):
    gray = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    faces = FACE.detectMultiScale(gray, 1.1, 4, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = [int(v) for v in sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[0]]
    return x, y, w, h

def auto_init_box(pil_img: Image.Image, size_label: str) -> tuple[int,int,int,int]:
    fb = detect_face_box(pil_img)
    W, H = pil_img.size
    w_t, h_t = map(int, size_label.split("x"))
    ar = w_t / float(h_t)
    if fb is None:
        # centered default
        win_h = H * 0.6
        win_w = win_h * ar
        cx, cy = W / 2.0, H / 2.0
    else:
        x, y, w, h = fb
        win_h = h * 1.7
        win_w = win_h * ar
        cx = x + w / 2.0
        cy = y + h / 2.0 + h * 0.15  # bias a bit downward
    L = int(max(0, cx - win_w / 2.0))
    T = int(max(0, cy - win_h / 2.0))
    R = int(min(W, cx + win_w / 2.0))
    B = int(min(H, cy + win_h / 2.0))
    return L, T, R, B

# ───────────────────────────────────────────────────────────────
# Session state helpers (persist per image/size)

def key_box(kind: str, preview_name: str, size_label: str) -> str:
    return f"box:{kind}:{preview_name}:{size_label}"

def clean_base(filename: str) -> str:
    base = os.path.splitext(filename)[0]
    if "-" in base:
        base = base.split("-")[0]
    return base

# ───────────────────────────────────────────────────────────────
# Sidebar controls

with st.sidebar:
    st.markdown("### Export sizes")
    avatar_sizes = st.multiselect("Avatar", ["256x256", "500x345"], default=["256x256", "500x345"])
    hero_sizes   = st.multiselect("Hero",   ["1200x1165", "1500x920"], default=["1200x1165", "1500x920"])

    st.markdown("---")
    show_frame = st.toggle("Show crop frame (preview)", value=True)
    do_straight = st.toggle("Auto-straighten (eyes)", value=True)

# ───────────────────────────────────────────────────────────────
# Tabs & upload

t1, t2 = st.tabs(["Avatars (front)", "Heroes (side)"])

for k in ["front_bufs", "side_bufs", "front_names", "side_names", "preview_kind", "preview_name"]:
    if k not in st.session_state:
        st.session_state[k] = [] if "bufs" in k or "names" in k else None

def _read_bytes(up):
    b = up.read()
    up.seek(0)
    return b

def buffer_uploads(files, which):
    if not files:
        return
    bufs, names = [], []
    for f in files:
        try:
            bufs.append(_read_bytes(f))
            names.append(f.name)
        except Exception as e:
            st.error(f"Failed to read {f.name}: {e}")
    if which == "front":
        st.session_state.front_bufs, st.session_state.front_names = bufs, names
    else:
        st.session_state.side_bufs, st.session_state.side_names = bufs, names

# ───────────────────────────────────────────────────────────────
# Core rendering (one cropper per selected size)

def render_croppers(kind: str):
    pn = st.session_state.preview_name
    if not pn:
        st.info("Select a preview image above first.")
        return

    if kind == "avatar":
        if pn not in st.session_state.front_names:
            return
        idx = st.session_state.front_names.index(pn)
        b = st.session_state.front_bufs[idx]
        sizes = avatar_sizes
    else:
        if pn not in st.session_state.side_names:
            return
        idx = st.session_state.side_names.index(pn)
        b = st.session_state.side_bufs[idx]
        sizes = hero_sizes

    base_img = load_img_from_bytes(b)
    if base_img is None:
        st.error("Failed to decode image.")
        return

    if do_straight:
        base_img = auto_straighten(base_img)

    if not sizes:
        st.info("Select at least one export size in the sidebar.")
        return

    # Build a larger working image so the crop box has room to pan/scale smoothly
    for s in sizes:
        w_out, h_out = map(int, s.split("x"))
        st.markdown(f"#### {s}")

        work = base_img.copy()
        work_w, work_h = max(900, w_out * 2), max(900, h_out * 2)
        work.thumbnail((work_w, work_h), Image.LANCZOS)

        # Initial crop box in original coords → map to working coords
        k = key_box(kind, pn, s)
        if k not in st.session_state:
            L, T, R, B = auto_init_box(base_img, s)
            sx = work.size[0] / base_img.size[0]
            sy = work.size[1] / base_img.size[1]
            st.session_state[k] = {
                "left": int(L * sx),
                "top": int(T * sy),
                "right": int(R * sx),
                "bottom": int(B * sy),
            }

        # Show cropper (fixed aspect ratio) — NOTE: image passed as first positional arg
        aspect = (w_out, h_out)   # must be a 2-element tuple
st_cropper(
    work.convert("RGB"),
    realtime_update=True,
    aspect_ratio=aspect,
    box_color='#FF2B2B',
    return_type='box',
    key=f"cropper_{kind}_{pn}_{s}",
)

        # If user moved the box, store it (working coords)
        if crop_box and all(k2 in crop_box for k2 in ("left", "top", "width", "height")):
            st.session_state[k] = {
                "left": int(crop_box["left"]),
                "top": int(crop_box["top"]),
                "right": int(crop_box["left"] + crop_box["width"]),
                "bottom": int(crop_box["top"] + crop_box["height"]),
            }

        # Build a *live* preview output using the box (map back to original)
        box = st.session_state[k]
        sx = base_img.size[0] / work.size[0]
        sy = base_img.size[1] / work.size[1]
        x0 = int(round(box["left"] * sx))
        y0 = int(round(box["top"] * sy))
        x1 = int(round(box["right"] * sx))
        y1 = int(round(box["bottom"] * sy))

        # Clamp
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(base_img.size[0], x1), min(base_img.size[1], y1)

        preview = base_img.crop((x0, y0, x1, y1)).resize((w_out, h_out), Image.LANCZOS)
        overlay = load_overlay(GUIDE_URLS.get(s, ""), w_out, h_out)
        if overlay:
            preview = Image.alpha_composite(preview.convert("RGBA"), overlay)
        if show_frame:
            preview = draw_frame(preview)

        # Slightly smaller preview for UI
        st.markdown('<div class="img-frame">', unsafe_allow_html=True)
        st.image(preview.resize((int(w_out * 0.75), int(h_out * 0.75)), Image.LANCZOS), use_column_width=False)
        st.markdown('</div>', unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────
# Export ZIP using saved boxes

def export_zip(bufs, names, sizes, label):
    if not bufs or not sizes:
        st.warning("Nothing to export.")
        return
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, b in zip(names, bufs):
            img = load_img_from_bytes(b)
            if img is None:
                continue
            if do_straight:
                img = auto_straighten(img)
            pn = name
            for s in sizes:
                w_out, h_out = map(int, s.split("x"))
                # If we have a stored working-box, map it back; else use auto-init
                k = key_box("avatar" if label == "avatar" else "hero", pn, s)
                if k in st.session_state:
                    work = img.copy()
                    work_w, work_h = max(900, w_out * 2), max(900, h_out * 2)
                    work.thumbnail((work_w, work_h), Image.LANCZOS)
                    sx = img.size[0] / work.size[0]
                    sy = img.size[1] / work.size[1]
                    box = st.session_state[k]
                    x0 = int(round(box["left"] * sx))
                    y0 = int(round(box["top"] * sy))
                    x1 = int(round(box["right"] * sx))
                    y1 = int(round(box["bottom"] * sy))
                else:
                    L, T, R, B = auto_init_box(img, s)
                    x0, y0, x1, y1 = L, T, R, B

                # Clamp & crop
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(img.size[0], x1), min(img.size[1], y1)
                out = img.crop((x0, y0, x1, y1)).resize((w_out, h_out), Image.LANCZOS)

                base = clean_base(name)
                buf = io.BytesIO()
                out.save(buf, format="PNG", optimize=True)  # transparent PNG
                zf.writestr(f"{base}-{label}_{w_out}x{h_out}.png", buf.getvalue())

    mem.seek(0)
    st.download_button(
        f"⬇️ Download {label.title()} ZIP",
        mem,
        file_name=f"{label}_images.zip",
        mime="application/zip",
    )

# ───────────────────────────────────────────────────────────────
# Avatars (front)
with t1:
    c1, c2 = st.columns([1, 2])
    with c1:
        files = st.file_uploader(
            "Upload front profile images",
            type=["tif", "tiff", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="front_upl",
        )
        if files:
            buffer_uploads(files, "front")
        if st.session_state.front_names:
            st.selectbox("Preview image", st.session_state.front_names, key="preview_name_front")
            st.session_state.preview_kind = "avatar"
            st.session_state.preview_name = st.session_state.get("preview_name_front")
        st.button("✅ Generate Avatars (ZIP)", key="gen_avatar_btn")
    with c2:
        if st.session_state.preview_kind == "avatar" and st.session_state.preview_name:
            render_croppers("avatar")

# Heroes (side)
with t2:
    c3, c4 = st.columns([1, 2])
    with c3:
        files = st.file_uploader(
            "Upload side profile images",
            type=["tif", "tiff", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="side_upl",
        )
        if files:
            buffer_uploads(files, "side")
        if st.session_state.side_names:
            st.selectbox("Preview image", st.session_state.side_names, key="preview_name_side")
            st.session_state.preview_kind = "hero"
            st.session_state.preview_name = st.session_state.get("preview_name_side")
        st.button("✅ Generate Heroes (ZIP)", key="gen_hero_btn")
    with c4:
        if st.session_state.preview_kind == "hero" and st.session_state.preview_name:
            render_croppers("hero")

# Export buttons
with t1:
    if st.session_state.get("gen_avatar_btn"):
        export_zip(st.session_state.front_bufs, st.session_state.front_names, avatar_sizes, "avatar")
with t2:
    if st.session_state.get("gen_hero_btn"):
        export_zip(st.session_state.side_bufs, st.session_state.side_names, hero_sizes, "hero")
