
import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw
import io, os, zipfile, sys, requests
from streamlit_drawable_canvas import st_canvas

# ──────────────────────────────────────────────────────────────────────────────
# Guide overlay URLs (GitHub raw)
GUIDE_URLS = {
    "256x256":  "https://raw.githubusercontent.com/lhkee/athlete-avatar-generator/ae2815490a0ad2861801054277022572b1a06eee/256x256-guide.png",
    "500x345":  "https://raw.githubusercontent.com/lhkee/athlete-avatar-generator/3936a686c65728e1811e9370fadd991125d91e61/500x345-guide.png",
    "1200x1165":"https://raw.githubusercontent.com/lhkee/athlete-avatar-generator/934f591fa238f6f597816307a6fc655792e15279/1200x1165-guide.png",
    "1500x920": "https://raw.githubusercontent.com/lhkee/athlete-avatar-generator/934f591fa238f6f597816307a6fc655792e15279/1500x920-guide.png",
}

st.set_page_config(page_title="Athlete Image Generator (Drag-to-Pan)", layout="wide")
st.title("🏋️ Athlete Image Generator — Drag-to-pan + scale + guides")

# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_overlay_from_url(url: str, target_w: int, target_h: int):
    if not url or not url.strip():
        return None
    try:
        r = requests.get(url.strip(), timeout=10)
        r.raise_for_status()
        g = Image.open(io.BytesIO(r.content)).convert("RGBA")
        if g.size != (target_w, target_h):
            g = g.resize((target_w, target_h), Image.BICUBIC)
        return g
    except Exception:
        return None

def _compose_with_overlay(img: Image.Image, overlay: Image.Image|None) -> Image.Image:
    base = img.convert("RGBA")
    if overlay is not None:
        base = Image.alpha_composite(base, overlay)
    return base

def _draw_frame(img: Image.Image) -> Image.Image:
    """1px red outer stroke + 2px inner mint stroke, always visible."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    # Outer red (true red)
    draw.rectangle([(0,0),(w-1,h-1)], outline=(255,0,0,255), width=1)
    # Inner mint
    if w>6 and h>6:
        draw.rectangle([(3,3),(w-4,h-4)], outline=(0,255,160,255), width=2)
    return img

def clean_base(filename):
    base = os.path.splitext(filename)[0]
    if "-" in base:
        base = base.split("-")[0]
    return base

# ──────────────────────────────────────────────────────────────────────────────
# IO helpers
@st.cache_data(show_spinner=False)
def load_img_from_bytes(b, max_dim=3000):
    try:
        img = Image.open(io.BytesIO(b))
        img.load()  # Pillow can read TIFF incl. LZW on Streamlit Cloud
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        w,h = img.size
        if max(w,h) > max_dim:
            s = max_dim/float(max(w,h))
            img = img.resize((int(w*s), int(h*s)), Image.LANCZOS)
        return img
    except Exception:
        return None

# Simple face box for initial fit
FRONTAL = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
def face_box(pil_img):
    gray = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    faces = FRONTAL.detectMultiScale(gray, 1.1, 4, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(60,60))
    if len(faces)==0:
        return None
    x,y,w,h = [int(v) for v in sorted(faces, key=lambda b: b[2]*b[3], reverse=True)[0]]
    return x,y,w,h

# ──────────────────────────────────────────────────────────────────────────────
# Session state utilities for per-image/size pan & zoom
def state_key(pn, s, kind): return f"{kind}:{pn}:{s}"
def init_state_for(pn, s, kind, img, auto_crop=None):
    k = state_key(pn,s,kind)
    if k in st.session_state: return
    W,H = img.size
    w,h = map(int, s.split("x"))
    if auto_crop:
        L,T,R,B = auto_crop
        cx = (L+R)/2.0; cy=(T+B)/2.0; win_h = (B-T)
    else:
        cx, cy = W/2.0, H/2.0; win_h = H*0.6
    st.session_state[k] = {"cx": float(cx), "cy": float(cy), "win_h": float(win_h)}

def update_from_drag(pn, s, kind, dx_canvas, dy_canvas, canvas_w, canvas_h):
    k = state_key(pn,s,kind)
    st_e = st.session_state.get(k)
    if st_e is None: return
    W,H = st.session_state["base_img"].size
    w,h = map(int, s.split("x"))
    win_h = st_e["win_h"]
    win_w = win_h * (w/float(h))
    # map canvas delta → original px
    scale_x = win_w / float(canvas_w)
    scale_y = win_h / float(canvas_h)
    st_e["cx"] = float(np.clip(st_e["cx"] - dx_canvas*scale_x, win_w/2, W - win_w/2))
    st_e["cy"] = float(np.clip(st_e["cy"] - dy_canvas*scale_y, win_h/2, H - win_h/2))

def apply_zoom(pn, s, kind, zoom_pct):
    k = state_key(pn,s,kind)
    st_e = st.session_state.get(k)
    if st_e is None: return
    if f"{k}:auto_h" in st.session_state:
        base = st.session_state[f"{k}:auto_h"]
    else:
        base = st_e["win_h"]
    st_e["win_h"] = float(np.clip(base * (1.0 + zoom_pct/100.0), 40, st.session_state["base_img"].size[1]))

def current_crop_box(pn, s, kind):
    k = state_key(pn,s,kind)
    st_e = st.session_state.get(k)
    if st_e is None: return None
    w,h = map(int, s.split("x"))
    cx, cy, win_h = st_e["cx"], st_e["cy"], st_e["win_h"]
    win_w = win_h * (w/float(h))
    L = int(round(cx - win_w/2)); T = int(round(cy - win_h/2))
    R = int(round(cx + win_w/2)); B = int(round(cy + win_h/2))
    W,H = st.session_state["base_img"].size
    L = max(0, L); T = max(0, T); R = min(W, R); B = min(H, B)
    return (L,T,R,B)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
with st.sidebar:
    st.markdown("### Export sizes")
    avatar_opts = st.multiselect("Avatar", ["256x256","500x345"], default=["256x256","500x345"])
    hero_opts   = st.multiselect("Hero",   ["1200x1165","1500x920"], default=["1200x1165","1500x920"])
    show_frame = st.toggle("Show crop frame (preview)", value=True)

# Upload & choose preview
t1, t2 = st.tabs(["Avatars (front)", "Heroes (side)"])
for k in ["front_bufs","side_bufs","front_names","side_names","preview_kind","preview_name"]:
    if k not in st.session_state: st.session_state[k] = [] if "bufs" in k or "names" in k else None

def _read_all_bytes(up): b=up.read(); up.seek(0); return b
def _buffer_uploads(files, which):
    if not files: return
    bufs, names = [], []
    for f in files:
        try: bufs.append(_read_all_bytes(f)); names.append(f.name)
        except Exception as e: st.error(f"Failed to read {f.name}: {e}")
    if which=="front":
        st.session_state.front_bufs, st.session_state.front_names = bufs, names
    else:
        st.session_state.side_bufs, st.session_state.side_names   = bufs, names

with t1:
    c1, c2 = st.columns([1,2])
    with c1:
        ff = st.file_uploader("Upload front profile images", type=["tif","tiff","png","jpg","jpeg"], accept_multiple_files=True, key="front_upl")
        if ff: _buffer_uploads(ff, "front")
        if st.session_state.front_names:
            st.selectbox("Preview image", st.session_state.front_names, key="preview_name_front")
            st.session_state.preview_kind = "avatar"; st.session_state.preview_name = st.session_state.get("preview_name_front")
        st.button("✅ Generate Avatars (ZIP)", key="gen_avatar_btn")
    with c2:
        avatar_placeholder = st.empty()

with t2:
    c3, c4 = st.columns([1,2])
    with c3:
        sf = st.file_uploader("Upload side profile images", type=["tif","tiff","png","jpg","jpeg"], accept_multiple_files=True, key="side_upl")
        if sf: _buffer_uploads(sf, "side")
        if st.session_state.side_names:
            st.selectbox("Preview image", st.session_state.side_names, key="preview_name_side")
            st.session_state.preview_kind = "hero"; st.session_state.preview_name = st.session_state.get("preview_name_side")
        st.button("✅ Generate Heroes (ZIP)", key="gen_hero_btn")
    with c4:
        hero_placeholder = st.empty()

# ──────────────────────────────────────────────────────────────────────────────
# Auto initial crop from face box
def auto_init(img, size_label):
    fb = face_box(img)
    if fb is None:
        W,H = img.size
        return (int(W*0.2), int(H*0.2), int(W*0.8), int(H*0.8))
    x,y,w,h = fb
    w_label, h_label = map(int, size_label.split("x"))
    ar = w_label/float(h_label)
    win_h = h*1.7  # include hair and chin
    win_w = win_h*ar
    cx = x + w/2; cy = y + h/2 + h*0.15
    L = int(max(0, cx - win_w/2)); T = int(max(0, cy - win_h/2))
    R = int(min(img.size[0], cx + win_w/2)); B = int(min(img.size[1], cy + win_h/2))
    return (L,T,R,B)

def render_guided(kind):
    pn = st.session_state.preview_name
    if not pn:
        st.info("Select a preview image above first."); return

    if kind=="avatar":
        if pn not in st.session_state.front_names: return
        idx = st.session_state.front_names.index(pn); b = st.session_state.front_bufs[idx]
        sizes = avatar_opts
    else:
        if pn not in st.session_state.side_names: return
        idx = st.session_state.side_names.index(pn); b = st.session_state.side_bufs[idx]
        sizes = hero_opts

    img = load_img_from_bytes(b)
    if img is None: st.error("Failed to decode image."); return
    st.session_state["base_img"] = img

    if not sizes:
        st.info("Select at least one export size in the sidebar."); return

    for s in sizes:
        w, h = map(int, s.split("x"))
        st.markdown(f"#### {s}")

        # Init per (image,size) once
        k = state_key(pn, s, kind)
        if k not in st.session_state:
            auto_LTRB = auto_init(img, s)
            init_state_for(pn, s, kind, img, auto_LTRB)
            st.session_state[f"{k}:auto_h"] = float(auto_LTRB[3]-auto_LTRB[1])

        # Scale slider (relative to auto window)
        zoom_pct = st.slider("Scale (±15%)", -15, 15, 0, key=f"zoom_{k}")
        apply_zoom(pn, s, kind, zoom_pct)

        # Current crop box → resized to target
        L,T,R,B = current_crop_box(pn, s, kind)
        crop = img.crop((L,T,R,B)).resize((w,h), Image.LANCZOS)

        # Overlay + frame
        overlay = _load_overlay_from_url(GUIDE_URLS.get(s,""), w, h)
        composed = _compose_with_overlay(crop, overlay)
        if st.session_state.get("show_frame", True) or True:
            composed = _draw_frame(composed)

        # Canvas backdrop (75% scale)
        CAN_W, CAN_H = int(round(w*0.75)), int(round(h*0.75))
        result = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=0,
            background_image=composed.resize((CAN_W, CAN_H), Image.LANCZOS),
            update_streamlit=True,
            height=CAN_H,
            width=CAN_W,
            drawing_mode="freedraw",
            key=f"canvas_{k}"
        )

        # Interpret last drawn path as a drag delta
        if result.json_data and "objects" in result.json_data and len(result.json_data["objects"])>0:
            last = result.json_data["objects"][-1]
            path = last.get("path", [])
            if len(path) >= 2:
                x0,y0 = path[0][1], path[0][2]
                x1,y1 = path[-1][1], path[-1][2]
                dx, dy = (x1-x0), (y1-y0)
                update_from_drag(pn, s, kind, dx, dy, CAN_W, CAN_H)

        st.caption("Drag inside the preview to pan. Use the Scale slider to zoom.")

def export_zip(bufs, names, sizes, label):
    if not bufs or not sizes:
        st.warning("Nothing to export."); return
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, b in zip(names, bufs):
            img = load_img_from_bytes(b)
            if img is None: continue
            pn = name
            for s in sizes:
                w,h = map(int, s.split("x"))
                k = state_key(pn, s, "avatar" if label=="avatar" else "hero")
                if k not in st.session_state:
                    auto_LTRB = auto_init(img, s)
                    init_state_for(pn, s, "avatar" if label=="avatar" else "hero", img, auto_LTRB)
                L,T,R,B = current_crop_box(pn, s, "avatar" if label=="avatar" else "hero")
                out = img.crop((L,T,R,B)).resize((w,h), Image.LANCZOS)
                base = clean_base(name)
                buf = io.BytesIO()
                out.save(buf, format="PNG", optimize=True)
                zf.writestr(f"{base}-{label}_{w}x{h}.png", buf.getvalue())
    mem.seek(0)
    st.download_button(f"⬇️ Download {label.title()} ZIP", mem, file_name=f"{label}_images.zip", mime="application/zip")

# ──────────────────────────────────────────────────────────────────────────────
# Render guided sections + export
with t1:
    if st.session_state.preview_kind == "avatar" and st.session_state.preview_name:
        with avatar_placeholder:
            render_guided("avatar")
with t2:
    if st.session_state.preview_kind == "hero" and st.session_state.preview_name:
        with hero_placeholder:
            render_guided("hero")

with t1:
    if st.session_state.get("gen_avatar_btn"):
        export_zip(st.session_state.front_bufs, st.session_state.front_names, avatar_opts, "avatar")
with t2:
    if st.session_state.get("gen_hero_btn"):
        export_zip(st.session_state.side_bufs, st.session_state.side_names, hero_opts, "hero")
