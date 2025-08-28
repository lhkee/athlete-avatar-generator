import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io, os, zipfile
import sys, platform

# ───────────────────────── Mediapipe (graceful import for Py3.11 env)
try:
    import mediapipe as mp
    MP_OK = True
except Exception:
    MP_OK = False
    mp = None

st.set_page_config(page_title="Athlete Avatar & Hero Generator", layout="centered")

st.title("🏋️ Athlete Avatar & Hero Generator")
st.caption("Landmark crop + improved hairstyle detection • Pillow-first TIFF (LZW) • Streamlit Free compatible (Python 3.11).")

# Small env pane to confirm versions at runtime (helps debugging deployments)
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
        f"Mediapipe:{mp_ver}"
    )
_env_panel()

# ───────────────────────── Controls
colA, colB = st.columns([1,1])
with colA:
    debug = st.toggle("Debug Mode", value=False, help="Show detailed processing logs in the UI.")
with colB:
    auto_straighten = st.toggle("Auto-straighten (eyes)", value=True,
                                help="Deskew tilted faces via landmark eye line (uses Mediapipe when available).")

# Uploaders
front_files = st.file_uploader(
    "Front profile images (for Avatars)",
    type=["tif","tiff","png","jpg","jpeg"], accept_multiple_files=True, key="front"
)
side_files  = st.file_uploader(
    "Side profile images (for Heroes)",
    type=["tif","tiff","png","jpg","jpeg"], accept_multiple_files=True, key="side"
)

# Export sizes
st.subheader("Export sizes")
c1, c2 = st.columns(2)
with c1:
    avatar_opts = st.multiselect("Avatar", ["256x256","500x345"], default=["256x256","500x345"])
with c2:
    hero_opts   = st.multiselect("Hero",   ["1200x1165","1500x920"], default=["1200x1165","1500x920"])

# Target line fractions (y / output_height)
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

def _log(msg: str):
    if debug: st.write(msg)

# ───────────────────────── Robust loader: Pillow-first TIFF (handles LZW)
def _load_image(upload):
    try:
        name = upload.name
        size_bytes = getattr(upload, "size", None)
        if debug:
            st.write(f"📥 File: {name} • Size: {size_bytes if size_bytes else 'unknown'} bytes")

        raw = upload.read()
        upload.seek(0)
        bio = io.BytesIO(raw)

        # Pillow-first (supports LZW)
        img = Image.open(bio)
        img.load()
        if debug: st.write("🧩 Decoder: Pillow")

        if img.mode != "RGBA":
            img = img.convert("RGBA")

        # Downscale huge images (longest side ≤ 2600 px) for stability
        max_dim = 2600
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            nw, nh = int(w * scale), int(h * scale)
            _log(f"🔧 Downscaling from {w}x{h} → {nw}x{nh}")
            img = img.resize((nw, nh), Image.LANCZOS)

        _log(f"🖼️ Loaded image: {img.size}, mode={img.mode}")
        return img
    except Exception as e:
        st.error(f"❌ Failed to load {upload.name}: {e}")
        return None

# ───────────────────────── Haar fallback (face box)
FRONTAL = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
PROFILE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

def _detect_face_box(pil_img):
    bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = FRONTAL.detectMultiScale(gray, 1.1, 4, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(60,60))
    if len(faces)==0:
        faces = PROFILE.detectMultiScale(gray, 1.1, 3, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(60,60))
    if len(faces)==0:
        gray_flipped = cv2.flip(gray, 1)
        faces_flip = PROFILE.detectMultiScale(gray_flipped, 1.1, 3, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(60,60))
        if len(faces_flip)>0:
            h,w = gray.shape
            faces = [(w-(x+wf), y, wf, hf) for (x,y,wf,hf) in faces_flip]
    if len(faces)==0:
        return None
    faces_sorted = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)
    x,y,w,h = [int(v) for v in faces_sorted[0]]
    return (x,y,w,h)

# ───────────────────────── Landmark helpers (only if Mediapipe is available)
if MP_OK:
    mp_face = mp.solutions.face_mesh
    LM_CHIN = 152
    LM_FOREHEAD = 10
    LM_EYES = [33, 133, 362, 263, 159, 386]  # corners + upper lids

    def _mediapipe_landmarks(pil_img):
        img = np.array(pil_img.convert("RGB"))
        with mp_face.FaceMesh(static_image_mode=True, refine_landmarks=True,
                              max_num_faces=1, min_detection_confidence=0.5) as fm:
            res = fm.process(img)
            if not res.multi_face_landmarks:
                return None
            lm = res.multi_face_landmarks[0].landmark
            H, W = img.shape[0], img.shape[1]
            pts = np.array([[lm_i.x * W, lm_i.y * H] for lm_i in lm], dtype=np.float32)
            return pts

    def _eye_chin_forehead_y(landmarks):
        eye_y = float(np.mean([landmarks[i,1] for i in LM_EYES]))
        chin_y = float(landmarks[LM_CHIN,1])
        forehead_y = float(landmarks[LM_FOREHEAD,1])
        eye_x = float(np.mean([landmarks[i,0] for i in LM_EYES]))
        return eye_y, chin_y, forehead_y, eye_x

    # ── IMPROVED: wide-band scan for hairstyle top ────────────────────────────
    def _scan_top_of_hair(pil_img, forehead_y, x_center, face_w, max_scan_frac=0.35):
        """
        Scan a wide band (±30% face width) above the forehead up to 35% of image height
        to detect the strongest hair boundary. Adds adaptive margin for tall/voluminous hair.
        """
        gray = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
        H, W = gray.shape

        band_half = int(0.30 * max(face_w, 1))
        x0, x1 = max(0, int(x_center - band_half)), min(W-1, int(x_center + band_half))

        start = int(max(forehead_y - max_scan_frac*H, 0))
        stop  = int(max(forehead_y - 2, 0))

        win = gray[:, x0:x1]
        grad = cv2.Sobel(win, cv2.CV_32F, 0, 1, ksize=3)           # vertical gradient
        gline = np.mean(np.abs(grad), axis=1)                      # average across band
        seg = gline[start:stop] if stop>start else gline[:1]

        if seg.size > 0:
            idx = int(np.argmax(seg))
            top_y = start + idx
        else:
            top_y = max(forehead_y - 0.08*H, 0)                    # conservative fallback

        # If span is large or texture is strong above forehead, extend margin to avoid cropping hair
        span = (forehead_y - top_y) / max(H,1)
        band = gray[max(0,int(top_y)):int(forehead_y), x0:x1]
        texture = float(np.var(band)) if band.size else 0.0
        if span > 0.18 or texture > 250.0:                         # tall/curly hair
            top_y = max(top_y - 0.04*H, 0)                         # push crop up a bit more

        return float(top_y)

    def _face_shape_adjustments(eye_y, chin_y, forehead_y):
        face_h = max(chin_y - eye_y, 1.0)
        upper_h = max(eye_y - forehead_y, 1.0)
        ratio = face_h / upper_h
        adj = dict(eye_bias=0.0, scale_bias=1.0, shift_bias=0.0)
        if ratio < 1.5:      # wider
            adj["eye_bias"] = -0.012
            adj["shift_bias"] = -0.015
        elif ratio > 2.2:    # longer
            adj["eye_bias"] = +0.015
            adj["scale_bias"] = 1.035
            adj["shift_bias"] = +0.012
        return adj

    def _deskew_by_eyes(pil_img, landmarks):
        pL = landmarks[33]; pR = landmarks[263]
        dy, dx = (pR[1]-pL[1]), (pR[0]-pL[0])
        if abs(dx) < 1e-3: return pil_img
        ang = np.degrees(np.arctan2(dy, dx))
        if abs(ang) < 1.5: return pil_img
        if debug: st.write(f"↩️ Auto-straighten: rotating {-ang:.2f}°")
        return pil_img.rotate(-ang, resample=Image.BICUBIC, expand=True)

    def _crop_from_landmarks(pil_img, face_box, landmarks, target_label, kind):
        tw, th = map(int, target_label.split("x"))
        ar = tw / float(th)
        (x,y,w,h) = face_box
        cx = x + w/2.0
        W_img, H_img = pil_img.size

        eye_y, chin_y, forehead_y, eye_x = _eye_chin_forehead_y(landmarks)
        # improved hair scan uses face width
        top_y = _scan_top_of_hair(pil_img, forehead_y, x_center=cx, face_w=w)

        adj = _face_shape_adjustments(eye_y, chin_y, forehead_y)
        tgt = TARGETS[kind][target_label].copy()
        tgt["eye"] = np.clip(tgt["eye"] + adj["eye_bias"], 0.05, 0.95)

        # scale from eye-chin spacing
        crop_h = (chin_y - eye_y) / max((tgt["chin"] - tgt["eye"]), 1e-6)
        crop_h *= adj["scale_bias"]

        crop_top = eye_y - tgt["eye"] * crop_h
        crop_top += adj["shift_bias"] * th  # interpret as fraction of output height

        # ensure hair not cropped (adaptive margin)
        margin = 0.014 * th
        if (top_y - crop_top) < margin:
            crop_top = max(0.0, top_y - margin)

        crop_bottom = crop_top + crop_h
        crop_w = crop_h * ar
        crop_left = cx - crop_w/2.0
        crop_right = crop_left + crop_w

        # clamp
        if crop_left < 0:
            crop_right -= crop_left; crop_left = 0
        if crop_top < 0:
            crop_bottom -= crop_top; crop_top = 0
        if crop_right > W_img:
            shift = crop_right - W_img
            crop_left -= shift; crop_right = W_img
        if crop_bottom > H_img:
            shift = crop_bottom - H_img
            crop_top -= shift; crop_bottom = H_img

        L = int(max(0, round(crop_left)))
        T = int(max(0, round(crop_top)))
        R = int(min(W_img, round(crop_right)))
        B = int(min(H_img, round(crop_bottom)))
        if R <= L+10 or B <= T+10:
            # simple fallback: pad around face box by aspect
            ar2 = tw/float(th)
            x0,y0,fw,fh = (x,y,w,h)
            ch2 = int(max(fh*1.6, 60))
            cw2 = int(ch2*ar2)
            cx0 = x0 + fw/2; cy0 = y0 + fh/2
            L = int(max(0, cx0 - cw2/2)); T = int(max(0, cy0 - ch2/2))
            R = int(min(W_img, L+cw2));    B = int(min(H_img, T+ch2))

        return pil_img.crop((L,T,R,B))

# ───────────────────────── Simple face-centered crop (used as fallback)
def _face_center_crop(pil_img, face_box, target_wh, kind):
    tw, th = target_wh
    ar = tw / float(th)
    (x,y,w,h) = face_box
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
    return pil_img.crop((left, top, right, bottom))

def _clean_base(filename):
    base = os.path.splitext(filename)[0]
    if "-" in base:
        base = base.split("-")[0]
    return base

# ───────────────────────── Pipeline (batch + ZIP)
def _process(files, sizes, label):
    if not files or not sizes:
        st.warning("⚠️ No files or sizes selected."); return

    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for up in files:
            st.write(f"📂 Processing: **{up.name}**")
            img = _load_image(up)
            if img is None: continue

            # Haar face box (always)
            face = _detect_face_box(img)
            if face is None:
                st.warning(f"⚠️ No face detected in {up.name}. Skipped."); continue

            # Landmarks (if mediapipe available)
            lms = None
            if MP_OK:
                try:
                    lms = _mediapipe_landmarks(img)
                except Exception as e:
                    lms = None
                    if debug: st.write(f"⚠️ Mediapipe landmark error: {e}")

            # Deskew if possible
            if auto_straighten and lms is not None:
                img = _deskew_by_eyes(img, lms)
                try:
                    lms = _mediapipe_landmarks(img)
                except Exception:
                    lms = None

            base = _clean_base(up.name)

            for s in sizes:
                kind = "avatar" if label=="avatar" else "hero"
                w, h = map(int, s.split("x"))
                if lms is not None and MP_OK:
                    crop = _crop_from_landmarks(img, face, lms, s, kind)
                else:
                    crop = _face_center_crop(img, face, (w,h), kind)

                out = crop.resize((w,h), Image.LANCZOS)

                if debug:
                    pr = np.array(out)
                    tgt = TARGETS.get(kind, {}).get(s, None)
                    if tgt is not None:
                        ey = int(tgt["eye"]  * h)
                        cy = int(tgt["chin"] * h)
                        cv2.line(pr, (0, ey), (w-1, ey), (0,0,255), 1)
                        cv2.line(pr, (0, cy), (w-1, cy), (0,0,255), 1)
                    st.image(pr, caption=f"Preview ({s})", use_column_width=True)

                buf = io.BytesIO()
                out.save(buf, format="PNG", optimize=True)
                out_name = f"{base}-{label}_{w}x{h}.png"
                zf.writestr(out_name, buf.getvalue())
                if debug: st.success(f"✅ Exported: {out_name}")

    mem_zip.seek(0)
    st.download_button(f"⬇️ Download {label.title()} ZIP", mem_zip,
                       file_name=f"{label}_images.zip", mime="application/zip")

# ───────────────────────── Actions
cA, cB = st.columns(2)
with cA:
    if st.button("✅ Generate Avatars"):
        _process(front_files, avatar_opts, "avatar")
with cB:
    if st.button("✅ Generate Hero Images"):
        _process(side_files, hero_opts, "hero")
