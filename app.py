import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io, os, zipfile, sys

# ──────────────────────────────────────────────────────────────────────────────
# Optional cropper (for manual drag mode)
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

# ──────────────────────────────────────────────────────────────────────────────
# UI setup
st.set_page_config(page_title="Athlete Image Generator", layout="wide")
st.title("🏋️ Athlete Image Generator — Auto + Manual (Sliders / Drag)")
st.caption("Hair-safe auto-crop with bottom chin safety • TIFF (LZW) via Pillow • Streamlit Cloud compatible (Py 3.11)")

# Manual mode selector
mode = st.radio("Manual Mode", ["Off (Auto only)", "Sliders (batch)", "Drag-to-crop (single)"], index=1, horizontal=True)

# General toggles
col0, col1, col2 = st.columns(3)
with col0:
    debug = st.toggle("Debug", value=False, help="Show eye/chin guides and hair-top line on previews")
with col1:
    auto_straighten = st.toggle("Auto-straighten (eyes)", value=True, help="Deskew using eye line when landmarks are available")
with col2:
    hair_safe = st.toggle("Hair-safe segmentation", value=True, help="Use MediaPipe Selfie Segmentation to protect tall/curly hair; falls back to gradient scan")

# Uploaders
left, right = st.columns(2)
with left:
    front_files = st.file_uploader(
        "Front profile images (for Avatars)",
        type=["tif","tiff","png","jpg","jpeg"],
        accept_multiple_files=True
    )
with right:
    side_files = st.file_uploader(
        "Side profile images (for Heroes)",
        type=["tif","tiff","png","jpg","jpeg"],
        accept_multiple_files=True
    )

# Sizes and target lines (y as fraction of output height)
st.subheader("Export sizes")
c1, c2 = st.columns(2)
with c1:
    avatar_opts = st.multiselect("Avatar", ["256x256","500x345"], default=["256x256","500x345"])
with c2:
    hero_opts   = st.multiselect("Hero",   ["1200x1165","1500x920"], default=["1200x1165","1500x920"])

TARGETS = {
    "avatar": {
        "256x256":  {"eye": 0.44, "chin": 0.832},
        "500x345":  {"eye": 0.44, "chin": 0.835},
    },
    "hero": {
        "1200x1165": {"eye": 0.35, "chin": 0.418},
        "1500x920":  {"eye": 0.35, "chin": 0.417},
    },
}

# Safety margins (fractions of **output** height unless noted)
HAIR_MARGIN      = {"avatar": 0.022, "hero": 0.018}  # min space above detected hair top
BOTTOM_MARGIN    = {"avatar": 0.024, "hero": 0.020}  # min space below chin
HAIR_TALL_BONUS  = {"avatar": 0.010, "hero": 0.006}  # extra headroom for very tall/afro hair

# Batch slider knobs (Option A)
if mode == "Sliders (batch)":
    with st.expander("Manual Adjust (applies to this run)"):
        man_scale       = st.slider("Scale bias (±8%)",          -8, 8, 0, help="Enlarge/reduce crop around eye–chin alignment")
        man_vshift      = st.slider("Vertical shift (px)",      -40, 40, 0, help="Move crop up/down after alignment")
        man_eye_bias    = st.slider("Eye target bias (±2.5%)",  -25, 25, 0, help="Shift eye target up/down (0.1% steps)")
        man_hair_margin = st.slider("Hair top margin (+px)",      0, 40, 0, help="Extra headroom above detected hair top")
        man_chin_margin = st.slider("Chin bottom margin (+px)",   0, 40, 0, help="Extra safety below chin")
else:
    man_scale = man_vshift = man_eye_bias = man_hair_margin = man_chin_margin = 0

# Sidebar: environment info
def _env_panel():
    try:
        import mediapipe as _mp
        mp_ver = _mp.__version__
    except Exception:
        mp_ver = "NOT INSTALLED"
    st.sidebar.markdown("### Environment")
    st.sidebar.code(
        f"Python:   {sys.version.split()[0]}\n"
        f"Streamlit:{st.__version__}\n"
        f"NumPy:    {np.__version__}\n"
        f"OpenCV:   {cv2.__version__}\n"
        f"Mediapipe:{mp_ver}\n"
        f"Cropper:  {'OK' if CROPPER_OK else 'N/A'}"
    )
_env_panel()

# ──────────────────────────────────────────────────────────────────────────────
# Helpers

def _log(msg: str):
    if debug: st.write(msg)

def load_img(upload):
    """
    Load image using Pillow (handles TIFF LZW), convert to RGBA, and downscale very large inputs
    to stabilize CPU/memory usage on free hosting.
    """
    try:
        raw = upload.read()
        upload.seek(0)
        img = Image.open(io.BytesIO(raw))
        img.load()  # force decode (fixes some TIF lazy-load cases)
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        # Downscale very large inputs for reliability
        max_dim = 2600
        w, h = img.size
        if max(w, h) > max_dim:
            s = max_dim / float(max(w, h))
            nw, nh = int(w*s), int(h*s)
            _log(f"🔧 Downscaling {w}x{h} → {nw}x{nh}")
            img = img.resize((nw, nh), Image.LANCZOS)
        return img
    except Exception as e:
        st.error(f"❌ Failed to load {upload.name}: {e}")
        return None

# Haar cascades (always present with cv2)
FRONTAL = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
PROFILE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

def face_box(pil_img):
    """
    Detect a face bounding box using Haar cascades (frontal / profile / mirrored profile).
    Returns (x,y,w,h) in pixel coords or None if not found.
    """
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

# Mediapipe (optional) — landmarks + hair-top estimation
if MP_OK:
    mp_face = mp.solutions.face_mesh
    mp_seg  = mp.solutions.selfie_segmentation
    LM_CHIN = 152
    LM_FOREHEAD = 10
    LM_EYES = [33, 133, 362, 263, 159, 386]  # corners + upper lids

    def mesh_landmarks(pil_img):
        img = np.array(pil_img.convert("RGB"))
        with mp_face.FaceMesh(static_image_mode=True, refine_landmarks=True,
                              max_num_faces=1, min_detection_confidence=0.5) as fm:
            res = fm.process(img)
            if not res.multi_face_landmarks:
                return None
            lm = res.multi_face_landmarks[0].landmark
            H, W = img.shape[0], img.shape[1]
            return np.array([[p.x*W, p.y*H] for p in lm], dtype=np.float32)

    def eye_chin_forehead_y(landmarks):
        eye_y = float(np.mean([landmarks[i,1] for i in LM_EYES]))
        chin_y = float(landmarks[LM_CHIN,1])
        forehead_y = float(landmarks[LM_FOREHEAD,1])
        eye_x = float(np.mean([landmarks[i,0] for i in LM_EYES]))
        return eye_y, chin_y, forehead_y, eye_x

    def hair_top_seg(pil_img, x_center, face_w, forehead_y):
        """
        Estimate top-of-hair using segmentation. Wide band, lower threshold for afros.
        Returns y (pixels) or None.
        """
        try:
            img_rgb = np.array(pil_img.convert("RGB"))
            H, W = img_rgb.shape[:2]
            with mp_seg.SelfieSegmentation(model_selection=1) as seg:
                res = seg.process(img_rgb)
                if res.segmentation_mask is None:
                    return None
                mask = res.segmentation_mask
                fg = (mask > 0.30).astype(np.uint8)  # lower threshold to catch wispy hair

            band_half = int(max(6, 0.40 * max(face_w, 1)))
            x0 = max(0, int(x_center) - band_half)
            x1 = min(W - 1, int(x_center) + band_half)
            band = fg[:, x0:x1]

            start = int(max(forehead_y - 0.55 * H, 0))
            stop  = int(max(forehead_y - 2, 0))
            if stop <= start:
                return None

            col_any = np.any(band[start:stop, :] > 0, axis=1)
            if not np.any(col_any):
                return None

            rows = np.flatnonzero(col_any)
            idx = int(np.percentile(rows, 1))  # very near the top, still robust to noise
            return float(start + idx)
        except Exception:
            return None

    def hair_top_grad(pil_img, forehead_y, x_center, face_w):
        """
        Fallback: vertical gradient scan to approximate top-of-hair.
        """
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
        if abs(dx) < 1e-3:
            return pil_img
        ang = np.degrees(np.arctan2(dy, dx))
        if abs(ang) < 1.5:
            return pil_img
        if debug: st.write(f"↩️ Auto-straighten: rotating {-ang:.2f}°")
        return pil_img.rotate(-ang, resample=Image.BICUBIC, expand=True)

def clean_base(filename):
    """
    Use original filename without '-' suffix numbers.
    Example: 'Maksim_Bakhtin-2890.tif' → 'Maksim_Bakhtin'
    """
    base = os.path.splitext(filename)[0]
    if "-" in base:
        base = base.split("-")[0]
    return base

def crop_landmarks(pil_img, face_box, landmarks, target_label, kind):
    """
    Landmark-based crop respecting target eye/chin lines, with hair-top and chin-bottom safety.
    Manual batch sliders are applied here too.
    """
    tw, th = map(int, target_label.split("x"))
    ar = tw / float(th)
    x,y,w,h = face_box
    cx = x + w/2.0
    W_img, H_img = pil_img.size

    # Landmarks → key ys
    eye_y, chin_y, forehead_y, eye_x = eye_chin_forehead_y(landmarks) if MP_OK else (None, None, None, None)

    # Hair top detection (segmentation first, then gradient)
    top_y = None
    if MP_OK and hair_safe:
        top_y = hair_top_seg(pil_img, x_center=cx, face_w=w, forehead_y=forehead_y)
    if top_y is None and MP_OK:
        top_y = hair_top_grad(pil_img, forehead_y, x_center=cx, face_w=w)
    if top_y is None:
        # If MP is not available, fallback to simple forehead offset
        top_y = max(0.0, (y - 0.10 * H_img))

    # Keep your original eye–chin geometry, then apply manual tweaks
    adj = face_shape_adj(eye_y, chin_y, forehead_y) if MP_OK else dict(eye_bias=0.0, scale_bias=1.0, shift_bias=0.0)
    tgt = TARGETS[kind][target_label].copy()
    tgt["eye"] = np.clip(tgt["eye"] + adj.get("eye_bias",0.0) + (man_eye_bias * 0.001), 0.05, 0.95)

    crop_h = (chin_y - eye_y) / max((tgt["chin"] - tgt["eye"]), 1e-6)
    crop_h *= adj.get("scale_bias", 1.0) * (1.0 + man_scale/100.0)

    crop_top = eye_y - tgt["eye"] * crop_h
    crop_top += adj.get("shift_bias", 0.0) * th + man_vshift

    # Hair-top safety (upper)
    hair_span = (forehead_y - top_y) / max(H_img, 1.0) if MP_OK else 0.0
    top_margin = HAIR_MARGIN[kind] * th
    if hair_span > 0.20:
        top_margin += HAIR_TALL_BONUS[kind] * th
    top_margin += man_hair_margin  # slider adds pixels
    if (top_y - crop_top) < top_margin:
        crop_top = max(0.0, top_y - top_margin)

    # Chin safety (lower)
    crop_bottom = crop_top + crop_h
    bottom_margin = BOTTOM_MARGIN[kind] * th + man_chin_margin
    delta = (chin_y + bottom_margin) - crop_bottom
    if delta > 0:
        crop_top += delta
        crop_bottom += delta

    # If upper & lower constraints conflict, allow gentle expansion
    if crop_top < 0:
        expand = -crop_top
        crop_h += expand
        crop_top = 0
        crop_bottom = crop_top + crop_h

    # Width by aspect
    crop_w = crop_h * ar
    crop_left = cx - crop_w/2.0
    crop_right = crop_left + crop_w

    # Clamp to image bounds
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

    # Extreme fallback
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

def crop_center(pil_img, face_box, target_wh, kind):
    """Simple face-centered fallback when landmarks aren't available."""
    tw, th = target_wh
    ar = tw / float(th)
    x,y,w,h = face_box
    cx, cy = x + w/2.0, y + h/2.0
    face_frac = 0.60 if kind=="avatar" else 0.52
    crop_h = max(int(h / face_frac), 60)
    crop_w = int(crop_h * ar)

    left = int(cx - crop_w/2.0)
    top  = int(cy - crop_h/2.0)
    right = left + crop_w
    bottom = top + crop_h

    W, H = pil_img.size
    if left < 0:
        right -= left; left = 0
    if top < 0:
        bottom -= top; top = 0
    if right > W:
        left -= (right - W); right = W
    if bottom > H:
        top -= (bottom - H); bottom = H

    left = max(left, 0); top = max(top, 0)
    right = min(right, W); bottom = min(bottom, H)
    return pil_img.crop((left, top, right, bottom)), None

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline

def process(files, sizes, label):
    if not files or not sizes:
        st.warning("⚠️ No files or sizes selected.")
        return

    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for up in files:
            st.write(f"📂 **{up.name}**")
            img = load_img(up)
            if img is None:
                continue

            f = face_box(img)
            if f is None:
                st.warning(f"⚠️ No face detected in {up.name}. Skipped.")
                continue

            # Landmarks (if available)
            lm = None
            if MP_OK:
                try:
                    lm = mesh_landmarks(img)
                except Exception as e:
                    lm = None
                    if debug: st.write(f"⚠️ Landmark error: {e}")

            # Auto-straighten (if possible)
            if auto_straighten and lm is not None and MP_OK:
                img = deskew_by_eyes(img, lm)
                try:
                    lm = mesh_landmarks(img)
                except Exception:
                    lm = None

            base = clean_base(up.name)

            # Option B: Drag-to-crop (single image recommended)
            manual_override = None
            if mode == "Drag-to-crop (single)":
                if not CROPPER_OK:
                    st.error("Drag mode requires streamlit-cropper. Add 'streamlit-cropper==0.2.2' to requirements.txt.")
                    return
                if len(files) == 1:
                    kind0 = "avatar" if label == "avatar" else "hero"
                    w0, h0 = map(int, sizes[0].split("x"))
                    # Preview image with guides
                    scale_w = min(900, img.size[0])
                    prev = img.convert("RGB").resize(
                        (scale_w, int(img.size[1] * (scale_w / img.size[0]))),
                        Image.LANCZOS
                    )
                    arr = np.array(prev).copy()
                    ey = int(TARGETS[kind0][sizes[0]]["eye"]  * arr.shape[0])
                    cy = int(TARGETS[kind0][sizes[0]]["chin"] * arr.shape[0])
                    cv2.line(arr, (0, ey), (arr.shape[1]-1, ey), (255, 0, 0), 1)  # eye (red)
                    cv2.line(arr, (0, cy), (arr.shape[1]-1, cy), (255, 0, 0), 1)  # chin (red)
                    prev_with_guides = Image.fromarray(arr)

                    rect = st_cropper(
                        prev_with_guides,
                        aspect_ratio=(w0 / h0),
                        box_color='#00E0AA',
                        return_type='box'
                    )
                    if rect and all(k in rect for k in ("left","top","width","height")):
                        scale = img.size[0] / prev.size[0]
                        L = int(rect['left']   * scale)
                        T = int(rect['top']    * scale)
                        R = int((rect['left'] + rect['width'])  * scale)
                        B = int((rect['top']  + rect['height']) * scale)
                        # Clamp to image bounds
                        L = max(0, min(L, img.size[0]-1))
                        T = max(0, min(T, img.size[1]-1))
                        R = max(L+1, min(R, img.size[0]))
                        B = max(T+1, min(B, img.size[1]))
                        manual_override = img.crop((L, T, R, B))
                    else:
                        st.warning("Could not read crop box; using auto-crop.")
                else:
                    st.warning("Drag mode is designed for a single image at a time. Using auto-crop for this batch.")

            # Export for each requested size
            for s in sizes:
                kind = "avatar" if label == "avatar" else "hero"
                w, h = map(int, s.split("x"))

                if manual_override is not None:
                    crop, dbg = manual_override, None
                elif lm is not None and MP_OK:
                    crop, dbg = crop_landmarks(img, f, lm, s, kind)
                else:
                    crop, dbg = crop_center(img, f, (w, h), kind)

                out = crop.resize((w, h), Image.LANCZOS)

                # Debug overlay
                if debug:
                    pr = np.array(out)
                    tgt = TARGETS[kind][s]
                    ey = int(tgt["eye"]  * h)
                    cy = int(tgt["chin"] * h)
                    cv2.line(pr, (0, ey), (w-1, ey), (255, 0, 0), 1)   # eye (red)
                    cv2.line(pr, (0, cy), (w-1, cy), (255, 0, 0), 1)   # chin (red)
                    if dbg and "top_y" in dbg and "crop_top" in dbg and "crop_bottom" in dbg:
                        # Map hair-top into output
                        scale_y = h / float((dbg["crop_bottom"] - dbg["crop_top"]) or 1.0)
                        hy = int((dbg["top_y"] - dbg["crop_top"]) * scale_y)
                        hy = np.clip(hy, 0, h-1)
                        cv2.line(pr, (0, hy), (w-1, hy), (0, 255, 255), 1)  # hair top (cyan)
                    st.image(pr, caption=f"Preview {s}", use_column_width=True)

                buf = io.BytesIO()
                out.save(buf, format="PNG", optimize=True)
                out_name = f"{base}-{label}_{w}x{h}.png"
                zf.writestr(out_name, buf.getvalue())

                if debug:
                    st.success(f"✅ Exported: {out_name}")

    mem_zip.seek(0)
    st.download_button(
        f"⬇️ Download {label.title()} ZIP",
        mem_zip,
        file_name=f"{label}_images.zip",
        mime="application/zip"
    )

# ──────────────────────────────────────────────────────────────────────────────
# Actions
cA, cB = st.columns(2)
with cA:
    if st.button("✅ Generate Avatars"):
        process(front_files, avatar_opts, "avatar")
with cB:
    if st.button("✅ Generate Heroes"):
        process(side_files, hero_opts, "hero")
