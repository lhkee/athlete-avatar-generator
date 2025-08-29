import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io, os, zipfile, sys, requests

# ──────────────────────────────────────────────────────────────────────────────
# Guide overlay URLs (GitHub raw)
GUIDE_URLS = {
    "256x256":  "https://raw.githubusercontent.com/lhkee/athlete-avatar-generator/ae2815490a0ad2861801054277022572b1a06eee/256x256-guide.png",
    "500x345":  "https://raw.githubusercontent.com/lhkee/athlete-avatar-generator/3936a686c65728e1811e9370fadd991125d91e61/500x345-guide.png",
    "1200x1165":"https://raw.githubusercontent.com/lhkee/athlete-avatar-generator/934f591fa238f6f597816307a6fc655792e15279/1200x1165-guide.png",
    "1500x920": "https://raw.githubusercontent.com/lhkee/athlete-avatar-generator/934f591fa238f6f597816307a6fc655792e15279/1500x920-guide.png",
}

# Fallback targets (used only if a guide fails to load)
TARGETS = {
    "avatar": {"256x256":{"eye":0.44,"chin":0.832}, "500x345":{"eye":0.44,"chin":0.835}},
    "hero":   {"1200x1165":{"eye":0.35,"chin":0.418}, "1500x920":{"eye":0.35,"chin":0.417}},
}

# Headroom/footroom cushions (applied on top of guide lines)
HAIR_MARGIN     = {"avatar":0.022, "hero":0.018}   # *th* units
BOTTOM_MARGIN   = {"avatar":0.024, "hero":0.020}

# ──────────────────────────────────────────────────────────────────────────────
# Page
st.set_page_config(page_title="Athlete Image Generator", layout="wide")
st.markdown("""
<style>
div.block-container{padding-top:1rem;padding-bottom:2rem;}
section[data-testid="stSidebar"] .stSlider,
section[data-testid="stSidebar"] .stMultiSelect,
section[data-testid="stSidebar"] .stRadio{margin-bottom:.35rem;}
.stButton>button{height:2.2rem;padding:0 .9rem;}
</style>
""", unsafe_allow_html=True)
st.title("🏋️ Athlete Image Generator — Auto + Guided tweak")

# ──────────────────────────────────────────────────────────────────────────────
# Overlays / guides / frame
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

def _composite_overlay(base_img: Image.Image, overlay_img: Image.Image) -> Image.Image:
    if base_img.mode != "RGBA":
        base_img = base_img.convert("RGBA")
    return Image.alpha_composite(base_img, overlay_img)

def _draw_frame_border(img_np: np.ndarray):
    """High-contrast double stroke (outer dark, inner mint)."""
    h, w = img_np.shape[:2]
    cv2.rectangle(img_np, (0, 0), (w-1, h-1), (20, 20, 20), 4, lineType=cv2.LINE_AA)     # outer
    cv2.rectangle(img_np, (3, 3), (w-4, h-4), (0, 255, 160), 2, lineType=cv2.LINE_AA)    # inner

def _find_red_lines(img_rgba: Image.Image):
    """
    Detect the three horizontal red lines in a guide PNG.
    Returns {'top': y, 'head': y, 'chin': y} (pixel rows) or None.
    """
    arr = np.array(img_rgba.convert("RGB"))
    H, W = arr.shape[:2]
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    red = (R > 180) & (G < 70) & (B < 70)

    counts = red.sum(axis=1)
    thresh = max(int(0.35 * W), 8)
    rows = np.where(counts >= thresh)[0]
    if rows.size == 0:
        return None

    lines, start, prev = [], rows[0], rows[0]
    for r in rows[1:]:
        if r == prev + 1:
            prev = r; continue
        lines.append((start, prev)); start = r; prev = r
    lines.append((start, prev))
    centers = sorted([int((a + b) / 2) for a, b in lines])

    if len(centers) < 2:
        return None
    if len(centers) > 3:
        top, bottom = centers[0], centers[-1]
        mid = min(centers[1:-1], key=lambda y: abs(y - (top + bottom) / 2))
        centers = [top, mid, bottom]

    top_y, head_y, chin_y = centers[0], centers[1], centers[-1]
    return {"top": top_y, "head": head_y, "chin": chin_y}

@st.cache_data(show_spinner=False)
def guide_targets_from_url(url: str, target_w: int, target_h: int):
    """Return {'eye': frac, 'chin': frac, 'top_frac': frac} (y/h fractions)."""
    ov = _load_overlay_from_url(url, target_w, target_h)
    if ov is None:
        return None
    ys = _find_red_lines(ov)
    if ys is None:
        return None
    h = float(target_h)
    return {"eye": ys["head"]/h, "chin": ys["chin"]/h, "top_frac": ys["top"]/h}

# ──────────────────────────────────────────────────────────────────────────────
# Optional Mediapipe (landmarks + segmentation)
try:
    import mediapipe as mp
    MP_OK = True
except Exception:
    MP_OK = False
    mp = None

# Sidebar controls
with st.sidebar:
    st.markdown("### Controls")
    mode = st.radio(
        "Manual Mode",
        ["Off (Auto only)", "Sliders (batch)", "Guided tweak (single)"],
        index=2
    )
    debug = st.toggle("Debug overlay (fallback)", value=False)
    auto_straighten = st.toggle("Auto-straighten (eyes)", value=True)
    hair_safe = st.toggle("Hair-safe segmentation", value=True)
    show_frame = st.toggle("Show crop frame border (preview)", value=True)

    st.markdown("### Export sizes")
    avatar_opts = st.multiselect("Avatar", ["256x256","500x345"], default=["256x256","500x345"])
    hero_opts   = st.multiselect("Hero",   ["1200x1165","1500x920"], default=["1200x1165","1500x920"])

    st.markdown("---")
    # Batch sliders (not used in Guided tweak)
    if mode == "Sliders (batch)":
        man_scale       = st.slider("Scale bias (±8%)",         -8, 8, 0)
        man_vshift      = st.slider("Vertical shift (px)",     -40, 40, 0)
        man_eye_bias    = st.slider("Eye target bias (±2.5%)", -25, 25, 0)
        man_hair_margin = st.slider("Hair margin (+px)",         0, 40, 0)
        man_chin_margin = st.slider("Chin margin (+px)",         0, 40, 0)
    else:
        man_scale = man_vshift = man_eye_bias = man_hair_margin = man_chin_margin = 0

    st.markdown("---")
    try:
        import mediapipe as _mp; mp_ver=_mp.__version__
    except Exception:
        mp_ver="NOT INSTALLED"
    st.code(
        f"Python: {sys.version.split()[0]}\n"
        f"Streamlit: {st.__version__}\n"
        f"NumPy: {np.__version__}\n"
        f"OpenCV: {cv2.__version__}\n"
        f"Mediapipe: {mp_ver}"
    )

# Session state
for k in ["front_bufs","side_bufs","front_names","side_names","preview_kind","preview_name"]:
    if k not in st.session_state:
        st.session_state[k] = [] if "bufs" in k or "names" in k else None

def _read_all_bytes(up): b = up.read(); up.seek(0); return b
def _buffer_uploads(files, which):
    if not files: return
    bufs, names = [], []
    for f in files:
        try:
            bufs.append(_read_all_bytes(f))
            names.append(f.name)
        except Exception as e:
            st.error(f"Failed to read {f.name}: {e}")
    if which=="front":
        st.session_state.front_bufs, st.session_state.front_names = bufs, names
    else:
        st.session_state.side_bufs, st.session_state.side_names   = bufs, names

# Uploaders + placeholders
t1, t2 = st.tabs(["Avatars (front)", "Heroes (side)"])
with t1:
    c1, c2 = st.columns([1,2])
    with c1:
        ff = st.file_uploader("Upload front profile images", type=["tif","tiff","png","jpg","jpeg"], accept_multiple_files=True, key="front_upl")
        if ff: _buffer_uploads(ff, "front")
        if st.session_state.front_names:
            st.selectbox("Preview image", st.session_state.front_names, key="preview_name_front")
            st.session_state.preview_kind = "avatar"
            st.session_state.preview_name = st.session_state.get("preview_name_front")
        st.button("✅ Generate Avatars (ZIP)", key="gen_avatar_btn")
    with c2:
        avatar_preview_placeholder = st.empty()

with t2:
    c3, c4 = st.columns([1,2])
    with c3:
        sf = st.file_uploader("Upload side profile images", type=["tif","tiff","png","jpg","jpeg"], accept_multiple_files=True, key="side_upl")
        if sf: _buffer_uploads(sf, "side")
        if st.session_state.side_names:
            st.selectbox("Preview image", st.session_state.side_names, key="preview_name_side")
            st.session_state.preview_kind = "hero"
            st.session_state.preview_name = st.session_state.get("preview_name_side")
        st.button("✅ Generate Heroes (ZIP)", key="gen_hero_btn")
    with c4:
        hero_preview_placeholder = st.empty()

# ──────────────────────────────────────────────────────────────────────────────
# Image IO / landmarks / detection

@st.cache_data(show_spinner=False)
def load_img_from_bytes(b, max_dim=2600):
    try:
        img = Image.open(io.BytesIO(b))
        img.load()  # Pillow can read TIFF (incl. LZW) without imagecodecs
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        w,h = img.size
        if max(w,h) > max_dim:
            s = max_dim/float(max(w,h))
            img = img.resize((int(w*s), int(h*s)), Image.LANCZOS)
        return img
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def mediapipe_landmarks_from_bytes(b):
    if not MP_OK: return None
    img = load_img_from_bytes(b)
    if img is None: return None
    arr = np.array(img.convert("RGB"))
    mp_face = mp.solutions.face_mesh
    with mp_face.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.5) as fm:
        res = fm.process(arr)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        H, W = arr.shape[0], arr.shape[1]
        return np.array([[p.x*W, p.y*H] for p in lm], dtype=np.float32)

FRONTAL = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
PROFILE = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_profileface.xml")

def face_box(pil_img):
    gray = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    faces = FRONTAL.detectMultiScale(gray, 1.1, 4, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(60,60))
    if len(faces) == 0:
        faces = PROFILE.detectMultiScale(gray, 1.1, 3, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(60,60))
    if len(faces) == 0:
        gray_flipped = cv2.flip(gray, 1)
        faces_flip = PROFILE.detectMultiScale(gray_flipped, 1.1, 3, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(60,60))
        if len(faces_flip) > 0:
            h, w = gray.shape
            faces = [(w-(x+wf), y, wf, hf) for (x,y,wf,hf) in faces_flip]
    if len(faces) == 0:
        return None
    x,y,w,h = [int(v) for v in sorted(faces, key=lambda b: b[2]*b[3], reverse=True)[0]]
    return (x,y,w,h)

# Landmark helpers
if MP_OK:
    mp_seg  = mp.solutions.selfie_segmentation
    LM_CHIN = 152
    LM_FOREHEAD = 10
    LM_EYES = [33,133,362,263,159,386]

    def eye_chin_forehead_y(landmarks):
        eye_y = float(np.mean([landmarks[i,1] for i in LM_EYES]))
        chin_y = float(landmarks[LM_CHIN,1])
        forehead_y = float(landmarks[LM_FOREHEAD,1])
        eye_x = float(np.mean([landmarks[i,0] for i in LM_EYES]))
        return eye_y, chin_y, forehead_y, eye_x

    def hair_top_seg(pil_img, x_center, face_w, forehead_y):
        try:
            img_rgb = np.array(pil_img.convert("RGB"))
            H, W = img_rgb.shape[:2]
            with mp_seg.SelfieSegmentation(model_selection=1) as seg:
                res = seg.process(img_rgb)
                if res.segmentation_mask is None:
                    return None
                mask = res.segmentation_mask
                fg = (mask > 0.30).astype(np.uint8)
            band_half = int(max(6, 0.40 * max(face_w, 1)))
            x0 = max(0, int(x_center) - band_half)
            x1 = min(W - 1, int(x_center) + band_half)
            band = fg[:, x0:x1]
            start = int(max(forehead_y - 0.55 * H, 0))
            stop  = int(max(forehead_y - 2, 0))
            if stop <= start: return None
            col_any = np.any(band[start:stop, :] > 0, axis=1)
            if not np.any(col_any): return None
            rows = np.flatnonzero(col_any)
            idx = int(np.percentile(rows, 1))
            return float(start + idx)
        except Exception:
            return None

    def hair_top_grad(pil_img, forehead_y, x_center, face_w):
        gray = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
        H, W = gray.shape
        band_half = int(0.30 * max(face_w, 1))
        x0, x1 = max(0, int(x_center - band_half)), min(W-1, int(x_center + band_half))
        start = int(max(forehead_y - 0.35 * H, 0))
        stop  = int(max(forehead_y - 2, 0))
        win = gray[:, x0:x1]
        grad = cv2.Sobel(win, cv2.CV_32F, 0, 1, ksize=3)
        gline = np.mean(np.abs(grad), axis=1)
        seg = gline[start:stop] if stop>start else gline[:1]
        if seg.size > 0:
            top_y = start + int(np.argmax(seg))
        else:
            top_y = max(forehead_y - 0.08*H, 0)
        span = (forehead_y - top_y) / max(H,1)
        band = gray[max(0,int(top_y)):int(forehead_y), x0:x1]
        texture = float(np.var(band)) if band.size else 0.0
        if span > 0.18 or texture > 250.0:
            top_y = max(top_y - 0.04*H, 0)
        return float(top_y)

    def face_shape_adj(eye_y, chin_y, forehead_y):
        face_h = max(chin_y - eye_y, 1.0)
        upper_h = max(eye_y - forehead_y, 1.0)
        ratio = face_h / upper_h
        adj = dict(eye_bias=0.0, scale_bias=1.0, shift_bias=0.0)
        if ratio < 1.5:      # wider faces
            adj["eye_bias"]   = -0.012
            adj["shift_bias"] = -0.015
        elif ratio > 2.2:    # longer faces
            adj["eye_bias"]   = +0.015
            adj["scale_bias"] = 1.035
            adj["shift_bias"] = +0.012
        return adj

    def deskew_by_eyes(pil_img, landmarks):
        pL = landmarks[33]; pR = landmarks[263]
        dy, dx = (pR[1]-pL[1]), (pR[0]-pL[0])
        if abs(dx) < 1e-3: return pil_img
        ang = np.degrees(np.arctan2(dy, dx))
        if abs(ang) < 1.5: return pil_img
        return pil_img.rotate(-ang, resample=Image.BICUBIC, expand=True)

# Cropping helpers
def clean_base(filename):
    base = os.path.splitext(filename)[0]
    if "-" in base: base = base.split("-")[0]
    return base

def crop_center(pil_img, face_box, target_wh, kind):
    tw, th = target_wh
    ar = tw / float(th)
    x,y,w,h = face_box
    cx, cy = x + w/2.0, y + h/2.0
    face_frac = 0.60 if kind=="avatar" else 0.52
    crop_h = max(int(h / face_frac), 60)
    crop_w = int(crop_h * ar)
    left = int(cx - crop_w/2.0); top  = int(cy - crop_h/2.0)
    right = left + crop_w;        bottom = top + crop_h
    W, H = pil_img.size
    if left < 0:   right -= left; left = 0
    if top < 0:    bottom -= top; top = 0
    if right > W:  left -= (right - W); right = W
    if bottom > H: top  -= (bottom - H); bottom = H
    left = max(left, 0); top = max(top, 0)
    right = min(right, W); bottom = min(bottom, H)
    return pil_img.crop((left, top, right, bottom)), None

def crop_landmarks(pil_img, face_box, landmarks, target_label, kind):
    tw, th = map(int, target_label.split("x"))
    ar = tw / float(th)
    x,y,w,h = face_box
    cx = x + w/2.0
    W_img, H_img = pil_img.size

    # landmarks
    eye_y, chin_y, forehead_y, eye_x = eye_chin_forehead_y(landmarks)

    # hair top (seg → grad → fallback)
    top_y = None
    if MP_OK and hair_safe:
        top_y = hair_top_seg(pil_img, x_center=cx, face_w=w, forehead_y=forehead_y)
    if top_y is None and MP_OK:
        top_y = hair_top_grad(pil_img, forehead_y, x_center=cx, face_w=w)
    if top_y is None:
        top_y = max(0.0, (y - 0.10 * H_img))

    # guide targets
    t_from_guide = guide_targets_from_url(GUIDE_URLS.get(target_label, ""), tw, th)
    if t_from_guide:
        tgt_eye  = float(t_from_guide["eye"])
        tgt_chin = float(t_from_guide["chin"])
        hair_top_target = float(t_from_guide["top_frac"])
    else:
        tgt = TARGETS[kind][target_label]
        tgt_eye, tgt_chin = float(tgt["eye"]), float(tgt["chin"])
        hair_top_target = 0.05

    # shape fine-tune
    adj = face_shape_adj(eye_y, chin_y, forehead_y) if MP_OK else dict(eye_bias=0.0, scale_bias=1.0, shift_bias=0.0)

    tgt_eye = np.clip(tgt_eye + adj.get("eye_bias",0.0) + (man_eye_bias * 0.001), 0.05, 0.95)

    crop_h = (chin_y - eye_y) / max((tgt_chin - tgt_eye), 1e-6)
    crop_h *= adj.get("scale_bias", 1.0) * (1.0 + man_scale/100.0)

    crop_top = eye_y - tgt_eye * crop_h
    crop_top += adj.get("shift_bias", 0.0) * th + man_vshift

    top_target_px = hair_top_target * crop_h
    need_margin = top_target_px + man_hair_margin + (HAIR_MARGIN[kind]*th)
    if (top_y - crop_top) < need_margin:
        crop_top = max(0.0, top_y - need_margin)

    crop_bottom = crop_top + crop_h
    bottom_margin = BOTTOM_MARGIN[kind] * th + man_chin_margin
    delta = (chin_y + bottom_margin) - crop_bottom
    if delta > 0:
        crop_top += delta
        crop_bottom += delta

    if crop_top < 0:
        expand = -crop_top
        crop_h += expand
        crop_top = 0
        crop_bottom = crop_top + crop_h

    crop_w = crop_h * ar
    crop_left = cx - crop_w/2.0
    crop_right = crop_left + crop_w

    # clamp
    if crop_left < 0:
        crop_right -= crop_left; crop_left = 0
    if crop_right > W_img:
        shift = crop_right - W_img
        crop_left -= shift; crop_right = W_img
    if crop_bottom > H_img:
        shift = crop_bottom - H_img
        crop_top -= shift; crop_bottom = H_img
    if crop_top < 0:
        crop_bottom -= crop_top; crop_top = 0

    L = int(max(0, round(crop_left)))
    T = int(max(0, round(crop_top)))
    R = int(min(W_img, round(crop_right)))
    B = int(min(H_img, round(crop_bottom)))

    if R <= L+10 or B <= T+10:
        # emergency fallback
        ar2 = tw/float(th)
        x0,y0,fw,fh = (x,y,w,h)
        ch2 = int(max(fh*1.6, 60))
        cw2 = int(ch2*ar2)
        cx0 = x0 + fw/2; cy0 = y0 + fh/2
        L = int(max(0, cx0 - cw2/2)); T = int(max(0, cy0 - ch2/2))
        R = int(min(W_img, L+cw2));    B = int(min(H_img, T+ch2))

    crop = pil_img.crop((L,T,R,B))
    dbg = {"crop_top": float(T), "crop_bottom": float(B), "top_y": float(top_y)}
    return crop, dbg

# ──────────────────────────────────────────────────────────────────────────────
# Previews & export

def _render_preview(kind, name):
    if not kind or not name: return
    if kind == "avatar":
        if name not in st.session_state.front_names: return
        idx = st.session_state.front_names.index(name)
        b = st.session_state.front_bufs[idx]
    else:
        if name not in st.session_state.side_names: return
        idx = st.session_state.side_names.index(name)
        b = st.session_state.side_bufs[idx]

    img = load_img_from_bytes(b)
    if img is None: st.error("Could not decode image."); return
    f = face_box(img)
    if f is None: st.error("No face detected."); return

    lm = mediapipe_landmarks_from_bytes(b) if MP_OK else None
    if auto_straighten and lm is not None and MP_OK:
        img = deskew_by_eyes(img, lm)

    sizes = avatar_opts if kind=="avatar" else hero_opts
    if not sizes:
        st.info("Select at least one export size in the sidebar."); return

    cols = st.columns(2) if len(sizes) > 1 else [st]
    ci = 0
    for s in sizes:
        w, h = map(int, s.split("x"))
        if lm is not None and MP_OK:
            crop, dbg = crop_landmarks(img, f, lm, s, kind)
        else:
            crop, dbg = crop_center(img, f, (w, h), kind)
        out = crop.resize((w, h), Image.LANCZOS)

        preview_img = out
        overlay = _load_overlay_from_url(GUIDE_URLS.get(s, ""), w, h)
        if overlay is not None:
            preview_img = _composite_overlay(preview_img, overlay)
            preview_np = np.array(preview_img)
        else:
            if debug:
                pr = np.array(preview_img)
                tgt = TARGETS[kind][s]
                ey = int(tgt["eye"]  * h)
                cy = int(tgt["chin"] * h)
                cv2.line(pr, (0, ey), (w-1, ey), (255, 0, 0), 1)
                cv2.line(pr, (0, cy), (w-1, cy), (255, 0, 0), 1)
                if dbg and "top_y" in dbg and "crop_top" in dbg and "crop_bottom" in dbg:
                    scale_y = h / float((dbg["crop_bottom"] - dbg["crop_top"]) or 1.0)
                    hy = int((dbg["top_y"] - dbg["crop_top"]) * scale_y)
                    hy = np.clip(hy, 0, h-1)
                    cv2.line(pr, (0, hy), (w-1, hy), (0, 255, 255), 1)
                preview_np = pr
            else:
                preview_np = np.array(preview_img)

        if show_frame:
            _draw_frame_border(preview_np)

        col = cols[ci]
        with col:
            st.image(preview_np, caption=f"Live preview • {s}", use_column_width=False, width=int(round(w*0.75)))
        ci = (ci + 1) % len(cols)

def _process_batch(bufs, names, sizes, label):
    if not bufs or not sizes:
        st.warning("No files or sizes selected."); return
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, b in zip(names, bufs):
            img = load_img_from_bytes(b)
            if img is None: st.warning(f"Skip {name}: decode error"); continue
            f = face_box(img)
            if f is None: st.warning(f"Skip {name}: no face"); continue
            lm = mediapipe_landmarks_from_bytes(b) if MP_OK else None
            if auto_straighten and lm is not None and MP_OK:
                img = deskew_by_eyes(img, lm)
            base = clean_base(name)
            for s in sizes:
                kind = "avatar" if label=="avatar" else "hero"
                w, h = map(int, s.split("x"))
                if lm is not None and MP_OK:
                    crop, _ = crop_landmarks(img, f, lm, s, kind)
                else:
                    crop, _ = crop_center(img, f, (w, h), kind)
                out = crop.resize((w, h), Image.LANCZOS)
                buf = io.BytesIO()
                out.save(buf, format="PNG", optimize=True)
                zf.writestr(f"{base}-{label}_{w}x{h}.png", buf.getvalue())
    mem.seek(0)
    st.download_button(f"⬇️ Download {label.title()} ZIP", mem, file_name=f"{label}_images.zip", mime="application/zip")

# Live preview
with t1:
    c1, c2 = st.columns([1,2])
with t1:
    if st.session_state.preview_kind == "avatar" and st.session_state.preview_name:
        with avatar_preview_placeholder:
            _render_preview("avatar", st.session_state.preview_name)
with t2:
    if st.session_state.preview_kind == "hero" and st.session_state.preview_name:
        with hero_preview_placeholder:
            _render_preview("hero", st.session_state.preview_name)

# Batch export actions
with t1:
    if st.session_state.get("gen_avatar_btn"):
        _process_batch(st.session_state.front_bufs, st.session_state.front_names, avatar_opts, "avatar")
with t2:
    if st.session_state.get("gen_hero_btn"):
        _process_batch(st.session_state.side_bufs, st.session_state.side_names, hero_opts, "hero")

# ──────────────────────────────────────────────────────────────────────────────
# Guided tweak (single) – per-image sliders above each preview
if mode == "Guided tweak (single)":
    st.markdown("---")
    st.subheader("Guided tweak (single image)")

    pk, pn = st.session_state.preview_kind, st.session_state.preview_name
    if not pk or not pn:
        st.info("Select a preview image above first.")
    else:
        if pk == "avatar":
            if pn not in st.session_state.front_names:
                st.warning("Preview image not found."); st.stop()
            idx = st.session_state.front_names.index(pn); b = st.session_state.front_bufs[idx]
            kind, sizes = "avatar", avatar_opts
        else:
            if pn not in st.session_state.side_names:
                st.warning("Preview image not found."); st.stop()
            idx = st.session_state.side_names.index(pn); b = st.session_state.side_bufs[idx]
            kind, sizes = "hero", hero_opts

        if not sizes:
            st.warning("Select at least one size in the sidebar.")
        else:
            img = load_img_from_bytes(b)
            if img is None:
                st.error("Could not decode image.")
            else:
                f = face_box(img)
                if f is None:
                    st.error("No face detected."); st.stop()
                lm = mediapipe_landmarks_from_bytes(b) if MP_OK else None
                base_img = img
                if auto_straighten and lm is not None and MP_OK:
                    base_img = deskew_by_eyes(img, lm)

                for s in sizes:
                    w, h = map(int, s.split("x"))

                    # Unique keys per preview
                    k_zoom  = f"zoom_{pn}_{s}"
                    k_vsh   = f"vshift_{pn}_{s}"

                    st.markdown(f"##### {s}")
                    cols_ctl = st.columns([1,1])
                    with cols_ctl[0]:
                        zoom_pct = st.slider("Zoom (±10%)", -10, 10, value=0, key=k_zoom)
                    with cols_ctl[1]:
                        vshift_px = st.slider("Vertical shift (±40 px)", -40, 40, value=0, key=k_vsh)

                    # 1) best auto-crop
                    if lm is not None and MP_OK:
                        crop, dbg = crop_landmarks(base_img, f, lm, s, kind)
                    else:
                        crop, dbg = crop_center(base_img, f, (w, h), kind)

                    # 2) apply the two tweaks (only to this preview)
                    if zoom_pct != 0 or vshift_px != 0:
                        cw, ch = crop.size
                        # vertical center from debug crop if available
                        if dbg and all(k in dbg for k in ("crop_top","crop_bottom")):
                            ct = int(dbg["crop_top"]); cb = int(dbg["crop_bottom"])
                            cy = (ct + cb) // 2
                        else:
                            cy = base_img.size[1] // 2
                        cx = base_img.size[0] // 2

                        # Zoom first (around current center)
                        if zoom_pct != 0:
                            factor = 1.0 + (zoom_pct / 100.0)
                            new_w = int(np.clip(cw * factor, 20, base_img.size[0]))
                            new_h = int(np.clip(ch * factor, 20, base_img.size[1]))
                            L = int(np.clip(cx - new_w//2, 0, base_img.size[0]-new_w))
                            T = int(np.clip(cy - new_h//2, 0, base_img.size[1]-new_h))
                            crop = base_img.crop((L, T, L+new_w, T+new_h))
                            cw, ch = crop.size

                        # Then vertical nudge
                        if vshift_px != 0:
                            cy = int(np.clip(cy + vshift_px, ch//2, base_img.size[1]-ch//2))
                            L = int(np.clip(cx - cw//2, 0, base_img.size[0]-cw))
                            T = int(np.clip(cy - ch//2, 0, base_img.size[1]-ch))
                            crop = base_img.crop((L, T, L+cw, T+ch))

                    out = crop.resize((w, h), Image.LANCZOS)

                    # overlay + frame for preview (75% size)
                    overlay = _load_overlay_from_url(GUIDE_URLS.get(s, ""), w, h)
                    if overlay is not None:
                        prev = np.array(_composite_overlay(out, overlay))
                    else:
                        prev = np.array(out)
                    if show_frame: _draw_frame_border(prev)

                    st.image(prev, caption=f"Guided preview • {s}", use_column_width=False, width=int(round(w*0.75)))
