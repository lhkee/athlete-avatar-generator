
import streamlit as st
import os
import cv2
from PIL import Image
import numpy as np
import zipfile
from io import BytesIO

st.set_page_config(page_title="Athlete Image Generator", layout="centered")
st.title("🏋️ Athlete Image Generator")
st.markdown("Upload front (avatar) or side (hero) profile images below. The app will export transparent PNGs in selected sizes.")

# Use OpenCV built-in path
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

if face_cascade.empty():
    raise IOError("🚫 OpenCV failed to load built-in haarcascade_frontalface_default.xml.")

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

def crop_to_face(image: Image.Image, target_size):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGR)
    faces = detect_face(img_cv)
    if len(faces) == 0:
        return image.resize(target_size)
    (x, y, w, h) = faces[0]
    cx, cy = x + w//2, y + h//2
    size = max(target_size)
    left = max(cx - size//2, 0)
    top = max(cy - size//2, 0)
    right = left + size
    bottom = top + size
    cropped = img_cv[top:bottom, left:right]
    cropped_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGBA))
    return cropped_img.resize(target_size)

def process_and_save(images, sizes, label):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        for img_file in images:
            filename = os.path.splitext(img_file.name)[0]
            base_name = filename.split("-")[0]
            for size in sizes:
                w, h = map(int, size.split("x"))
                image = Image.open(img_file).convert("RGBA")
                resized = crop_to_face(image, (w, h))
                out_filename = f"{base_name}-{label}_{w}x{h}.png"
                buffer = BytesIO()
                resized.save(buffer, format="PNG")
                zipf.writestr(out_filename, buffer.getvalue())
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
    zip_data = process_and_save(avatar_images, avatar_sizes, "avatar")
    st.download_button("Download Avatars ZIP", zip_data, "avatars.zip", mime="application/zip")

if st.button("✅ Generate Hero Images") and hero_images:
    zip_data = process_and_save(hero_images, hero_sizes, "hero")
    st.download_button("Download Hero ZIP", zip_data, "heroes.zip", mime="application/zip")
