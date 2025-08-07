
import streamlit as st
import os
import cv2
from PIL import Image
import numpy as np
import zipfile
from io import BytesIO
import traceback

st.set_page_config(page_title="Athlete Image Generator", layout="centered")
st.title("🏋️ Athlete Image Generator")
st.markdown("Upload front (avatar) or side (hero) profile images below. The app will export transparent PNGs in selected sizes.")

# Use OpenCV built-in path
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    st.error("🚫 Failed to load face cascade. Ensure OpenCV is installed correctly.")
    st.stop()

def detect_face(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        st.write(f"🔍 Detected {len(faces)} face(s)")
        return faces
    except Exception as e:
        st.error(f"Face detection error: {e}")
        st.text(traceback.format_exc())
        return []

def crop_to_face(image: Image.Image, target_size):
    try:
        st.write(f"Original image size: {image.size}, mode: {image.mode}")
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGR)
        faces = detect_face(img_cv)
        if len(faces) == 0:
            st.warning("⚠️ No face detected, resizing directly")
            return image.resize(target_size)
        (x, y, w, h) = faces[0]
        cx, cy = x + w//2, y + h//2
        size = max(target_size)
        left = max(cx - size//2, 0)
        top = max(cy - size//2, 0)
        right = left + size
        bottom = top + size
        st.write(f"Cropping area: {(left, top, right, bottom)}")
        cropped = img_cv[top:bottom, left:right]
        cropped_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGBA))
        return cropped_img.resize(target_size)
    except Exception as e:
        st.error(f"Cropping error: {e}")
        st.text(traceback.format_exc())
        return image

def process_and_save(images, sizes, label):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        for img_file in images:
            try:
                st.write(f"📂 Processing: {img_file.name}")
                filename = os.path.splitext(img_file.name)[0]
                base_name = filename.split("-")[0]
                image = Image.open(img_file).convert("RGBA")
                for size in sizes:
                    w, h = map(int, size.split("x"))
                    resized = crop_to_face(image, (w, h))
                    out_filename = f"{base_name}-{label}_{w}x{h}.png"
                    buffer = BytesIO()
                    resized.save(buffer, format="PNG")
                    zipf.writestr(out_filename, buffer.getvalue())
                    st.success(f"✅ Created: {out_filename}")
            except Exception as e:
                st.error(f"Processing error for {img_file.name}: {e}")
                st.text(traceback.format_exc())
    return zip_buffer.getvalue()

# Upload UI
avatar_images = st.file_uploader("📤 Upload Front Profile Image(s) for Avatar", accept_multiple_files=True, type=["png", "jpg", "jpeg", "tif", "tiff"])
hero_images = st.file_uploader("📤 Upload Side Profile Image(s) for Hero", accept_multiple_files=True, type=["png", "jpg", "jpeg", "tif", "tiff"])

# Size selectors
st.subheader("🛠 Export Sizes (Default checked)")
avatar_sizes = st.multiselect("Avatar Sizes", ["256x256", "500x345"], default=["256x256", "500x345"])
hero_sizes = st.multiselect("Hero Sizes", ["1200x1165", "1500x920"], default=["1200x1165", "1500x920"])

# Process buttons
if st.button("✅ Generate Avatars") and avatar_images:
    try:
        st.info("🔄 Generating avatar images...")
        zip_data = process_and_save(avatar_images, avatar_sizes, "avatar")
        st.download_button("⬇️ Download Avatars ZIP", zip_data, "avatars.zip", mime="application/zip")
    except Exception as e:
        st.error(f"Unhandled error: {e}")
        st.text(traceback.format_exc())

if st.button("✅ Generate Hero Images") and hero_images:
    try:
        st.info("🔄 Generating hero images...")
        zip_data = process_and_save(hero_images, hero_sizes, "hero")
        st.download_button("⬇️ Download Hero ZIP", zip_data, "heroes.zip", mime="application/zip")
    except Exception as e:
        st.error(f"Unhandled error: {e}")
        st.text(traceback.format_exc())
