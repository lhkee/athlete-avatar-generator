import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io, os, zipfile

st.set_page_config(page_title="Athlete Avatar & Hero Generator", layout="centered")

st.title("🏋️ Athlete Avatar & Hero Generator")
st.caption("Upload front (avatars) and side (heroes) images. Pillow-first TIFF decode (LZW), free-tier friendly.")

# ───────────────────────── Controls
colA, colB = st.columns([1,1])
with colA:
    debug = st.toggle("Debug Mode", value=False, help="Show detailed processing logs in the UI.")
with colB:
    auto_straighten = st.toggle("Auto-straighten (eyes)", value=True,
                                help="Deskew tilted faces via Haar eye detection.")

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

# ───────────────────────── Haar cascades (built-in; no external files)
FRONTAL = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
PROFILE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
EYES    = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def _log(msg: str):
    if debug:
        st.write(msg)

# ───────────────────────── Robust loader: Pillow-first TIFF (handles LZW)
def _load_image(upload):
    """Use Pillow to decode TIFF/PNG/JPG. Works with LZW TIFF (no imagecodecs).
       Preserves transparency. Downscales very large images for memory stability."""
    try:
        name = upload.name
        size_bytes = getattr(upload, "size", None)
        if debug:
            st.write(f"📥 File: {name} • Size: {size_bytes if size_bytes else 'unknown'} bytes")

        # Read raw bytes once; allow Streamlit to reuse by rewinding.
        raw = upload.read()
        upload.seek(0)
        bio = io.BytesIO(raw)

        # Pillow-first (supports LZW)
        try:
            img = Image.open(bio)
            img.load()  # force decode
            if debug: st.write("🧩 Decoder: Pillow")
        except Exception as e_pil:
            st.error(f"❌ Failed to load {name} via Pillow: {e_pil}")
            return None

        # Normalize to RGBA
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        # Downscale super large images (longest side ≤ 2600 px) to keep memory usage in check
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

# ───────────────────────── Face + eyes (for roll) detection
def _detect_faces_cv2(pil_img):
    """Return list of (x,y,w,h) via Haar cascades. Try frontal → profile → mirrored profile."""
    try:
        bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        faces = FRONTAL.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4,
            flags=cv2.CASCADE_SCALE_IMAGE, minSize=(60,60)
        )

        # Convert to list of tuples if any
        faces_list = [tuple(map(int, b)) for b in faces] if len(faces) else []

        if len(faces_list) == 0:
            faces_p = PROFILE.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3,
                flags=cv2.CASCADE_SCALE_IMAGE, minSize=(60,60)
            )
            faces_list = [tuple(map(int, b)) for b in faces_p] if len(faces_p) else []

        if len(faces_list) == 0:
            # try mirrored (detect right profile by flipping)
            gray_flipped = cv2.flip(gray, 1)
            faces_flip = PROFILE.detectMultiScale(
                gray_flipped, scaleFactor=1.1, minNeighbors=3,
                flags=cv2.CASCADE_SCALE_IMAGE, minSize=(60,60)
            )
            if len(faces_flip) > 0:
                h, w = gray.shape
                mapped = []
                for (x,y,wf,hf) in faces_flip:
                    mapped.append((w - (x+wf), y, wf, hf))
                faces_list = mapped

        return faces_list  # always a Python list
    except Exception as e:
        st.error(f"❌ Face detection error: {e}")
        return []

def _estimate_roll_from_eyes(pil_img):
    """Estimate tilt angle using eye positions; return degrees (negative = clockwise)."""
    try:
        bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        eyes = EYES.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE, minSize=(20,20)
        )
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])  # leftmost, rightmost
            (x1,y1,w1,h1) = eyes[0]
            (x2,y2,w2,h2) = eyes[-1]
            p1 = np.array([x1 + w1/2.0, y1 + h1/2.0])
            p2 = np.array([x2 + w2/2.0, y2 + h2/2.0])
            dy = p2[1] - p1[1]
            dx = p2[0] - p1[0]
            if dx == 0:
                return 0.0
            angle = np.degrees(np.arctan2(dy, dx))
            return angle
    except Exception:
        pass
    return 0.0

def _rotate_image(pil_img, angle):
    if abs(angle) < 2.0:  # ignore tiny jitter
        return pil_img
    if debug: st.write(f"↩️ Auto-straighten: rotating {-angle:.2f}°")
    return pil_img.rotate(-angle, resample=Image.BICUBIC, expand=True)

# ───────────────────────── Face-centered crop with calibrated margin
def _face_center_crop(pil_img, face_box, target_wh, kind):
    """Crop around face with calibrated margin so face size looks consistent; enforce target aspect."""
    tw, th = target_wh
    ar = tw / float(th)
    (x,y,w,h) = face_box
    cx, cy = x + w/2.0, y + h/2.0

    # Heuristics to match the manual references:
    if kind == "avatar":
        face_frac = 0.60   # face ≈ 60% of crop height
    else:  # hero
        face_frac = 0.52   # slightly wider framing

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
    # drop trailing "-####" if present
    if "-" in base:
        base = base.split("-")[0]
    return base

# ───────────────────────── Batch process + ZIP
def _process_files(files, sizes, label):
    if not files or not sizes:
        st.warning("⚠️ No files or sizes selected.")
        return

    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for up in files:
            st.write(f"📂 Processing: **{up.name}**")
            img = _load_image(up)
            if img is None:
                continue

            if auto_straighten:
                angle = _estimate_roll_from_eyes(img)
                img = _rotate_image(img, angle)

            faces = _detect_faces_cv2(img)
            if debug: st.write(f"🔍 Faces detected: {len(faces)}")
            if faces is None or len(faces) == 0:
                st.warning(f"⚠️ No face detected in {up.name}. Export skipped.")
                continue

            # choose largest face
            faces_sorted = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)
            face = tuple(int(v) for v in faces_sorted[0])

            base = _clean_base(up.name)

            for s in sizes:
                w, h = map(int, s.split("x"))
                kind = "avatar" if label == "avatar" else "hero"
                crop = _face_center_crop(img, face, (w,h), kind)
                out  = crop.resize((w,h), Image.LANCZOS)

                b = io.BytesIO()
                out.save(b, format="PNG", optimize=True)
                out_name = f"{base}-{label}_{w}x{h}.png"
                zf.writestr(out_name, b.getvalue())
                if debug: st.success(f"✅ Exported: {out_name}")

    mem_zip.seek(0)
    st.download_button(
        f"⬇️ Download {label.title()} ZIP",
        mem_zip, file_name=f"{label}_images.zip", mime="application/zip"
    )

# ───────────────────────── Actions
cA, cB = st.columns(2)
with cA:
    if st.button("✅ Generate Avatars"):
        _process_files(front_files, avatar_opts, "avatar")
with cB:
    if st.button("✅ Generate Hero Images"):
        _process_files(side_files, hero_opts, "hero")
