
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io, os, zipfile
import tifffile

st.set_page_config(page_title="Athlete Avatar & Hero Generator", layout="centered")

st.title("🏋️ Athlete Avatar & Hero Generator")
st.caption("Drop front (avatars) and side (heroes) images. Works offline on Streamlit Free.")

# ---- Controls
colA, colB = st.columns([1,1])
with colA:
    debug = st.toggle("Debug Mode", value=False, help="Show detailed processing logs in the UI.")
with colB:
    auto_straighten = st.toggle("Auto-straighten (eyes)", value=True, help="Try to deskew tilted faces using eye detection.")

# ---- Uploaders
front_files = st.file_uploader("Front profile images (for Avatars)", type=["tif","tiff","png","jpg","jpeg"], accept_multiple_files=True, key="front")
side_files  = st.file_uploader("Side profile images (for Heroes)",  type=["tif","tiff","png","jpg","jpeg"], accept_multiple_files=True, key="side")

# ---- Sizes
st.subheader("Export sizes")
c1, c2 = st.columns(2)
with c1:
    avatar_opts = st.multiselect("Avatar", ["256x256","500x345"], default=["256x256","500x345"])
with c2:
    hero_opts   = st.multiselect("Hero",   ["1200x1165","1500x920"], default=["1200x1165","1500x920"])

# ---- Haar cascades (built-in: no external files)
FRONTAL = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
PROFILE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
EYES    = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def _log(msg):
    if debug: st.write(msg)

def _load_image(upload):
    """Read file (TIFF > 50MB ok), convert to RGBA, optionally downscale huge images to keep memory low."""
    try:
        size_bytes = getattr(upload, "size", None)
        name = upload.name
        if debug: st.write(f"📥 File: {name} • Size: {size_bytes if size_bytes else 'unknown'} bytes")
        if name.lower().endswith((".tif",".tiff")):
            import numpy as np
            arr = tifffile.imread(upload)
            # Preserve alpha if present
            if arr.ndim == 2:
                img = Image.fromarray(arr, mode="L").convert("RGBA")
            elif arr.ndim == 3 and arr.shape[2] == 4:
                img = Image.fromarray(arr.astype(np.uint8), mode="RGBA")
            else:
                img = Image.fromarray(arr.astype(np.uint8)).convert("RGBA")
        else:
            img = Image.open(upload)
            if img.mode != "RGBA":
                img = img.convert("RGBA")

        # Downscale super large images to keep memory under control (longest side ≤ 2600 px)
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

def _detect_faces_cv2(pil_img):
    """Return list of (x,y,w,h) via Haar cascades. Try frontal → profile → rotated profile."""
    try:
        bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = FRONTAL.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(60,60))
        if len(faces) == 0:
            faces = PROFILE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(60,60))
        if len(faces) == 0:
            gray_flipped = cv2.flip(gray, 1)
            faces_flip = PROFILE.detectMultiScale(gray_flipped, scaleFactor=1.1, minNeighbors=3, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(60,60))
            if len(faces_flip) > 0:
                h, w = gray.shape
                mapped = []
                for (x,y,wf,hf) in faces_flip:
                    mapped.append((w - (x+wf), y, wf, hf))
                faces = np.array(mapped, dtype=np.int32)
        return faces if len(faces) else []
    except Exception as e:
        st.error(f"❌ Face detection error: {e}")
        return []

def _estimate_roll_from_eyes(pil_img):
    """Estimate tilt angle using eye positions; return degrees (negative = clockwise)."""
    try:
        bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        eyes = EYES.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(20,20))
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])
            (x1,y1,w1,h1) = eyes[0]
            (x2,y2,w2,h2) = eyes[-1]
            p1 = np.array([x1 + w1/2.0, y1 + h1/2.0])
            p2 = np.array([x2 + w2/2.0, y2 + h2/2.0])
            dy = p2[1] - p1[1]
            dx = p2[0] - p1[0]
            if dx == 0: return 0.0
            angle = np.degrees(np.arctan2(dy, dx))
            return angle
    except Exception:
        pass
    return 0.0

def _rotate_image(pil_img, angle):
    if abs(angle) < 2.0:
        return pil_img
    if debug: st.write(f"↩️ Auto-straighten: rotating {-angle:.2f}°")
    return pil_img.rotate(-angle, resample=Image.BICUBIC, expand=True)

def _face_center_crop(pil_img, face_box, target_wh, kind):
    tw, th = target_wh
    ar = tw / float(th)
    (x,y,w,h) = face_box
    cx, cy = x + w/2.0, y + h/2.0

    if kind == "avatar":
        face_frac = 0.60
    else:
        face_frac = 0.52

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
            if not faces:
                st.warning(f"⚠️ No face detected in {up.name}. Export skipped.")
                continue

            faces_sorted = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)
            face = tuple(int(v) for v in faces_sorted[0])

            base = os.path.splitext(up.name)[0].split("-")[0]

            for s in sizes:
                w, h = map(int, s.split("x"))
                kind = "avatar" if label == "avatar" else "hero"
                crop = _face_center_crop(img, face, (w,h), kind)
                out = crop.resize((w,h), Image.LANCZOS)

                b = io.BytesIO()
                out.save(b, format="PNG", optimize=True)
                out_name = f"{base}-{label}_{w}x{h}.png"
                zf.writestr(out_name, b.getvalue())
                if debug: st.success(f"✅ Exported: {out_name}")

    mem_zip.seek(0)
    st.download_button(f"⬇️ Download {label.title()} ZIP", mem_zip, file_name=f"{label}_images.zip", mime="application/zip")

# ---- Actions
cA, cB = st.columns(2)
with cA:
    if st.button("✅ Generate Avatars"):
        _process_files(front_files, avatar_opts, "avatar")
with cB:
    if st.button("✅ Generate Hero Images"):
        _process_files(side_files, hero_opts, "hero")
