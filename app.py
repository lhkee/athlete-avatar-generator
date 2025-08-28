
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io, os, zipfile
import mediapipe as mp

st.set_page_config(page_title="Athlete Avatar & Hero Generator", layout="centered")

st.title("🏋️ Athlete Avatar & Hero Generator")
st.caption("Per-image landmark crop • Pillow-first TIFF (LZW) • Streamlit Free compatible")

# ───────────────────────── Controls
colA, colB = st.columns([1,1])
with colA:
    debug = st.toggle("Debug Mode", value=False, help="Show detailed processing logs in the UI.")
with colB:
    auto_straighten = st.toggle("Auto-straighten (eyes)", value=True,
                                help="Deskew tilted faces via landmark eye line.")

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

# ───────────────────────── Targets (fractions of output height)
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

# ───────────────────────── MediaPipe Face Mesh (landmarks)
mp_face = mp.solutions.face_mesh
LM_CHIN = 152
LM_FOREHEAD = 10
LM_EYES = [33, 133, 362, 263, 159, 386]  # corners + upper lids

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

# ───────────────────────── Landmarks & helpers
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

def _scan_top_of_hair(pil_img, forehead_y, x_center, max_scan_px=0.22):
    gray = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    H, W = gray.shape
    col = int(np.clip(x_center, 0, W-1))
    start = int(max(forehead_y - int(max_scan_px*H), 0))
    stop  = int(max(forehead_y - 2, 0))
    win = gray[:, max(0,col-4):min(W,col+5)]
    grad = cv2.Sobel(win, cv2.CV_32F, 0, 1, ksize=3)
    gline = np.mean(np.abs(grad), axis=1)
    seg = gline[start:stop] if stop>start else gline[:1]
    if seg.size > 0:
        idx = int(np.argmax(seg))
        return float(start + idx)
    return max(forehead_y - 0.06*H, 0.0)

def _face_shape_adjustments(eye_y, chin_y, forehead_y):
    face_h = max(chin_y - eye_y, 1.0)
    upper_h = max(eye_y - forehead_y, 1.0)
    ratio = face_h / upper_h

    adj = dict(eye_bias=0.0, scale_bias=1.0, shift_bias=0.0)

    # Wider faces → eyes slightly above target, shift up a bit
    if ratio < 1.5:
        adj["eye_bias"] = -0.012
        adj["shift_bias"] = -0.015

    # Longer faces → enlarge slightly, shift down, eyes below target
    elif ratio > 2.2:
        adj["eye_bias"] = +0.015
        adj["scale_bias"] = 1.035
        adj["shift_bias"] = +0.012

    return adj

def _hair_type(pil_img, forehead_y, top_y, x_center):
    gray = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    H, W = gray.shape
    col0, col1 = int(max(0,x_center-20)), int(min(W-1,x_center+20))
    y0, y1 = int(max(0, top_y)), int(min(H-1, forehead_y))
    if y1 <= y0: return "normal"
    patch = gray[y0:y1, col0:col1]
    var = float(np.var(patch))
    span = (forehead_y - top_y) / max(H,1)
    if var < 80:     return "bald"
    if span > 0.14:  return "tall"
    return "normal"

def _deskew_by_eyes(pil_img, landmarks):
    # angle from two outer eye corners (approx: 33 and 263)
    pL = landmarks[33]; pR = landmarks[263]
    dy, dx = (pR[1]-pL[1]), (pR[0]-pL[0])
    if abs(dx) < 1e-3: return pil_img
    ang = np.degrees(np.arctan2(dy, dx))
    if abs(ang) < 1.5: return pil_img
    if debug: st.write(f"↩️ Auto-straighten: rotating {-ang:.2f}°")
    return pil_img.rotate(-ang, resample=Image.BICUBIC, expand=True)

# ───────────────────────── Crop from landmarks
def _crop_from_landmarks(pil_img, face_box, landmarks, target_label, kind):
    tw, th = map(int, target_label.split("x"))
    ar = tw / float(th)
    (x,y,w,h) = face_box
    cx = x + w/2.0
    W_img, H_img = pil_img.size

    eye_y, chin_y, forehead_y, eye_x = _eye_chin_forehead_y(landmarks)
    top_y = _scan_top_of_hair(pil_img, forehead_y, x_center=cx)

    adj = _face_shape_adjustments(eye_y, chin_y, forehead_y)
    tgt = TARGETS[kind][target_label].copy()
    tgt["eye"] = np.clip(tgt["eye"] + adj["eye_bias"], 0.05, 0.95)

    # vertical scale from eye-chin spacing
    crop_h = (chin_y - eye_y) / max((tgt["chin"] - tgt["eye"]), 1e-6)
    crop_h *= adj["scale_bias"]

    crop_top = eye_y - tgt["eye"] * crop_h
    crop_top += adj["shift_bias"] * th

    # respect top-of-hair (with small margin)
    margin = 0.012 * th
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
        # Fallback: center on face box
        return pil_img.crop((max(0,x- int(0.2*w)),
                             max(0,y- int(0.2*h)),
                             min(W_img, x+ int(1.2*w)),
                             min(H_img, y+ int(1.2*h))))

    return pil_img.crop((L,T,R,B))

# ───────────────────────── Haar fallback (for face box) 
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

            lms = _mediapipe_landmarks(img)
            if lms is None:
                st.warning(f"⚠️ No landmarks found in {up.name}. Trying Haar fallback.")
            face = _detect_face_box(img)
            if face is None:
                st.warning(f"⚠️ No face detected in {up.name}. Skipped.")
                continue

            # Optional deskew
            if auto_straighten and lms is not None:
                img = _deskew_by_eyes(img, lms)
                # re-detect landmarks after rotation
                lms = _mediapipe_landmarks(img)

            base = _clean_base(up.name)

            for s in sizes:
                kind = "avatar" if label=="avatar" else "hero"
                w, h = map(int, s.split("x"))
                if lms is not None:
                    crop = _crop_from_landmarks(img, face, lms, s, kind)
                else:
                    # last resort: enlarge around face, keep aspect
                    ar = w/float(h)
                    x,y,fw,fh = face
                    ch = int(max(fh*1.6, 60))
                    cw = int(ch*ar)
                    cx = x + fw/2
                    cy = y + fh/2
                    L = int(max(0, cx - cw/2)); T = int(max(0, cy - ch/2))
                    R = int(min(img.size[0], L+cw)); B = int(min(img.size[1], T+ch))
                    crop = img.crop((L,T,R,B))

                out = crop.resize((w,h), Image.LANCZOS)

                # Optional debug overlay of target lines
                if debug and lms is not None:
                    pr = np.array(out)
                    tgt = TARGETS[kind][s]
                    ey = int(tgt["eye"]  * h)
                    cy = int(tgt["chin"] * h)
                    cv2.line(pr, (0, ey), (w-1, ey), (0,0,255), 1)
                    cv2.line(pr, (0, cy), (w-1, cy), (0,0,255), 1)
                    st.image(pr, caption=f"Debug preview + targets · {s}", use_column_width=True)

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
