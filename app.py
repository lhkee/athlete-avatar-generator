import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io, os, zipfile, sys, requests

# ─────────────────────────────────────────────
# Hard-coded guide overlay URLs (GitHub raw)
GUIDE_URLS = {
    "256x256":  "https://raw.githubusercontent.com/lhkee/athlete-avatar-generator/ae2815490a0ad2861801054277022572b1a06eee/256x256-guide.png",
    "500x345":  "https://raw.githubusercontent.com/lhkee/athlete-avatar-generator/3936a686c65728e1811e9370fadd991125d91e61/500x345-guide.png",
    "1200x1165":"https://raw.githubusercontent.com/lhkee/athlete-avatar-generator/934f591fa238f6f597816307a6fc655792e15279/1200x1165-guide.png",
    "1500x920": "https://raw.githubusercontent.com/lhkee/athlete-avatar-generator/934f591fa238f6f597816307a6fc655792e15279/1500x920-guide.png",
}

# ─────────────────────────────────────────────
# Overlay + frame helpers
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

def _draw_frame_border(img_np: np.ndarray, color=(255,255,255), thickness=1):
    h, w = img_np.shape[:2]
    cv2.rectangle(img_np, (0,0), (w-1,h-1), color, thickness)

# ─────────────────────────────────────────────
# Optional cropper (manual drag mode)
try:
    from streamlit_cropper import st_cropper
    CROPPER_OK = True
except Exception:
    CROPPER_OK = False

# Optional Mediapipe (landmarks + segmentation for hair-safe)
try:
    import mediapipe as mp
    MP_OK = True
except Exception:
    MP_OK = False
    mp = None

# ─────────────────────────────────────────────
# Page + compact layout
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
st.title("🏋️ Athlete Image Generator — Auto + Manual")

# Targets and safety constants
TARGETS = {
    "avatar": {"256x256":{"eye":0.44,"chin":0.832}, "500x345":{"eye":0.44,"chin":0.835}},
    "hero":   {"1200x1165":{"eye":0.35,"chin":0.418}, "1500x920":{"eye":0.35,"chin":0.417}},
}
HAIR_MARGIN = {"avatar":0.022,"hero":0.018}
BOTTOM_MARGIN={"avatar":0.024,"hero":0.020}
HAIR_TALL_BONUS={"avatar":0.010,"hero":0.006}

# Sidebar controls
with st.sidebar:
    st.markdown("### Controls")
    mode = st.radio("Manual Mode", ["Off (Auto only)", "Sliders (batch)", "Drag-to-crop (single)"], index=1)
    debug = st.toggle("Debug overlay (fallback)", value=False)
    auto_straighten = st.toggle("Auto-straighten (eyes)", value=True)
    hair_safe = st.toggle("Hair-safe segmentation", value=True)
    st.markdown("---")
    show_frame = st.toggle("Show crop frame border (preview)", value=True)

    st.markdown("### Export sizes")
    avatar_opts = st.multiselect("Avatar", ["256x256","500x345"], default=["256x256","500x345"])
    hero_opts   = st.multiselect("Hero",   ["1200x1165","1500x920"], default=["1200x1165","1500x920"])

    st.markdown("---")
    if mode == "Sliders (batch)":
        st.markdown("### Manual Adjust (batch)")
        man_scale       = st.slider("Scale bias (±8%)",         -8, 8, 0)
        man_vshift      = st.slider("Vertical shift (px)",     -40, 40, 0)
        man_eye_bias    = st.slider("Eye target bias (±2.5%)", -25, 25, 0)
        man_hair_margin = st.slider("Hair margin (+px)",         0, 40, 0)
        man_chin_margin = st.slider("Chin margin (+px)",         0, 40, 0)
        st.caption(
            f"Active: scale={man_scale:+}%, vshift={man_vshift:+} px, "
            f"eye_bias={man_eye_bias:+}‰, hair+={man_hair_margin}px, chin+={man_chin_margin}px"
        )
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
        f"Mediapipe: {mp_ver}\n"
        f"Cropper: {'OK' if CROPPER_OK else 'N/A'}"
    )

# Session state to persist uploads
for k in ["front_bufs","side_bufs","front_names","side_names","preview_kind","preview_name"]:
    if k not in st.session_state:
        st.session_state[k] = [] if "bufs" in k or "names" in k else None

def _read_all_bytes(up):
    b = up.read(); up.seek(0); return b

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

# Cached heavy steps
@st.cache_data(show_spinner=False)
def load_img_from_bytes(b, max_dim=2600):
    try:
        img = Image.open(io.BytesIO(b))
        img.load()  # Pillow handles TIFF (incl. LZW)
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

# Haar cascades
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
        if ratio < 1.5:
            adj["eye_bias"]   = -0.012
            adj["shift_bias"] = -0.015
        elif ratio > 2.2:
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

# Crop helpers
def clean_base(filename):
    base = os.path.splitext(filename)[0]
    if "-" in base:
        base = base.split("-")[0]
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

    eye_y, chin_y, forehead_y, eye_x = eye_chin_forehead_y(landmarks)
    # Hair top (seg → grad → fallback)
    top_y = None
    if MP_OK and hair_safe:
        top_y = hair_top_seg(pil_img, x_center=cx, face_w=w, forehead_y=forehead_y)
    if top_y is None and MP_OK:
        top_y = hair_top_grad(pil_img, forehead_y, x_center=cx, face_w=w)
    if top_y is None:
        top_y = max(0.0, (y - 0.10 * H_img))

    adj = face_shape_adj(eye_y, chin_y, forehead_y) if MP_OK else dict(eye_bias=0.0, scale_bias=1.0, shift_bias=0.0)
    tgt = TARGETS[kind][target_label].copy()
    tgt["eye"] = np.clip(tgt["eye"] + adj.get("eye_bias",0.0) + (man_eye_bias * 0.001), 0.05, 0.95)

    crop_h = (chin_y - eye_y) / max((tgt["chin"] - tgt["eye"]), 1e-6)
    crop_h *= adj.get("scale_bias", 1.0) * (1.0 + man_scale/100.0)

    crop_top = eye_y - tgt["eye"] * crop_h
    crop_top += adj.get("shift_bias", 0.0) * th + man_vshift

    hair_span = (forehead_y - top_y) / max(H_img, 1.0)
    top_margin = HAIR_MARGIN[kind] * th
    if hair_span > 0.20:
        top_margin += HAIR_TALL_BONUS[kind] * th
    top_margin += man_hair_margin
    if (top_y - crop_top) < top_margin:
        crop_top = max(0.0, top_y - top_margin)

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

# Preview & export
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
        img = deskew_by_eyes(img, lm)  # preview acceptable without recompute

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

        # Compose: base -> guide (URL) or debug -> frame -> show exact pixels
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
            _draw_frame_border(preview_np, (255,255,255), 1)

        with cols[ci]:
            st.image(preview_np, caption=f"Live preview • {s}", use_column_width=False, width=w)
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

# Live preview (persistent)
with t1:
    if st.session_state.preview_kind == "avatar" and st.session_state.preview_name:
        with avatar_preview_placeholder:
            _render_preview("avatar", st.session_state.preview_name)
with t2:
    if st.session_state.preview_kind == "hero" and st.session_state.preview_name:
        with hero_preview_placeholder:
            _render_preview("hero", st.session_state.preview_name)

# Actions
with t1:
    if st.session_state.get("gen_avatar_btn"):
        _process_batch(st.session_state.front_bufs, st.session_state.front_names, avatar_opts, "avatar")
with t2:
    if st.session_state.get("gen_hero_btn"):
        _process_batch(st.session_state.side_bufs, st.session_state.side_names, hero_opts, "hero")

# Drag-to-crop (single) — aspect_ratio tuple fix
if mode == "Drag-to-crop (single)":
    st.markdown("---")
    st.subheader("Drag-to-crop (single image)")
    if not CROPPER_OK:
        st.error("Drag mode requires `streamlit-cropper==0.2.2` in requirements.txt.")
    else:
        pk, pn = st.session_state.preview_kind, st.session_state.preview_name
        if not pk or not pn:
            st.info("Select a preview image above first.")
        else:
            if pk == "avatar":
                if pn not in st.session_state.front_names: st.warning("Preview image not found."); st.stop()
                idx = st.session_state.front_names.index(pn); b = st.session_state.front_bufs[idx]
                kind, sizes = "avatar", avatar_opts
            else:
                if pn not in st.session_state.side_names: st.warning("Preview image not found."); st.stop()
                idx = st.session_state.side_names.index(pn); b = st.session_state.side_bufs[idx]
                kind, sizes = "hero", hero_opts

            if not sizes:
                st.warning("Select at least one size in the sidebar.")
            else:
                img = load_img_from_bytes(b)
                if img is None:
                    st.error("Could not decode image.")
                else:
                    w0, h0 = map(int, sizes[0].split("x"))
                    prev_w = min(900, img.size[0])
                    prev = img.convert("RGB").resize((prev_w, int(img.size[1]*(prev_w/img.size[0]))), Image.LANCZOS)
                    arr = np.array(prev).copy()
                    tgt = TARGETS[kind][sizes[0]]
                    ey = int(tgt["eye"]  * arr.shape[0]); cy = int(tgt["chin"] * arr.shape[0])
                    cv2.line(arr,(0,ey),(arr.shape[1]-1,ey),(255,0,0),1)
                    cv2.line(arr,(0,cy),(arr.shape[1]-1,cy),(255,0,0),1)
                    prev2 = Image.fromarray(arr)

                    # Overlay the size-specific guide in drag preview too
                    url = GUIDE_URLS.get(f"{w0}x{h0}", "")
                    if url:
                        ov = _load_overlay_from_url(url, prev2.size[0], prev2.size[1])
                        if ov:
                            prev2 = Image.alpha_composite(prev2.convert("RGBA"), ov)

                    if not (w0>0 and h0>0): w0,h0 = 1,1
                    rect = st_cropper(
                        prev2,
                        aspect_ratio=(float(w0), float(h0)),  # tuple, not float
                        box_color='#00E0AA',
                        return_type='box',
                        key=f"crop_box_{pn.replace('.', '_')}"
                    )

                    if rect and all(k in rect for k in ("left","top","width","height")):
                        scale = img.size[0]/prev.size[0]
                        L=int(rect['left']*scale); T=int(rect['top']*scale)
                        R=int((rect['left']+rect['width'])*scale); B=int((rect['top']+rect['height'])*scale)
                        L = max(0, min(L, img.size[0]-1)); T = max(0, min(T, img.size[1]-1))
                        R = max(L+1, min(R, img.size[0]));   B = max(T+1, min(B, img.size[1]))
                        manual = img.crop((L,T,R,B))
                        st.success("Manual crop locked for this image and will be used when you export.")
                        for s in sizes:
                            w,h = map(int, s.split("x"))
                            out = manual.resize((w,h), Image.LANCZOS)
                            # show same overlay + frame in manual previews
                            overlay = _load_overlay_from_url(GUIDE_URLS.get(s, ""), w, h)
                            if overlay is not None:
                                outp = np.array(_composite_overlay(out, overlay))
                            else:
                                outp = np.array(out)
                            if show_frame: _draw_frame_border(outp, (255,255,255), 1)
                            st.image(outp, caption=f"Manual Preview • {pn} • {s}", use_column_width=False, width=w)
                    else:
                        st.warning("Drag a box over the image to set the crop.")
