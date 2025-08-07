import os
import io
import zipfile
import streamlit as st
import numpy as np
from PIL import Image
import cv2

st.set_page_config(page_title="Athlete Image Generator", layout="centered")

st.title("🏋️ Athlete Image Generator")
st.markdown("Upload front (avatar) or side (hero) profile images below. The app will export transparent PNGs in selected sizes.")

# Upload images
avatar_files = st.file_uploader("📤 Upload Front Profile Images", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True, key="front")
hero_files = st.file_uploader("📤 Upload Side Profile Images", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True, key="side")

avatar_sizes = ["256x256", "500x345"]
hero_sizes = ["1200x1165", "1500x920"]

selected_avatar_sizes = st.multiselect("Avatar Export Sizes", avatar_sizes, default=avatar_sizes)
selected_hero_sizes = st.multiselect("Hero Export Sizes", hero_sizes, default=hero_sizes)

generated_images = {}

# Load local haarcascade
HAAR_PATH = "haarcascade_frontalface_default.xml"
if not os.path.exists(HAAR_PATH):
    st.error("Missing haarcascade XML file. Please include haarcascade_frontalface_default.xml.")
    st.stop()

face_cascade = cv2.CascadeClassifier(HAAR_PATH)

def detect_and_crop_face(pil_img):
    try:
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
        if len(faces) == 0:
            return pil_img  # fallback: return original

        x, y, w, h = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)[0]
        buffer = int(0.6 * h)
        left = max(0, x - buffer)
        top = max(0, y - buffer)
        right = min(pil_img.width, x + w + buffer)
        bottom = min(pil_img.height, y + h + buffer)
        return pil_img.crop((left, top, right, bottom))
    except Exception as e:
        st.error(f"Face detection failed: {e}")
        return pil_img

def export_image(pil_img, size_str):
    w, h = map(int, size_str.split("x"))
    return pil_img.resize((w, h), Image.Resampling.LANCZOS)

def add_to_zip(zip_buffer, filename, image):
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG", optimize=True)
    zip_buffer.writestr(filename, img_bytes.getvalue())

def process_images(files, sizes, label):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        for file in files:
            try:
                base = file.name.rsplit(".", 1)[0]
                name = base.split("-")[0]
                img = Image.open(file).convert("RGBA")
                cropped = detect_and_crop_face(img)

                for size in sizes:
                    export = export_image(cropped, size)
                    filename = f"{name}-{label}_{size}.png"
                    if name not in generated_images:
                        generated_images[name] = []
                    generated_images[name].append((filename, export))
                    add_to_zip(zipf, filename, export)
            except Exception as e:
                st.error(f"Failed to process {file.name}: {e}")
    return zip_buffer

if avatar_files and st.button("🎨 Generate Avatars"):
    zip_data = process_images(avatar_files, selected_avatar_sizes, "avatar")
    st.download_button("⬇️ Download All Avatars", data=zip_data.getvalue(), file_name="avatars.zip", mime="application/zip")

if hero_files and st.button("🎨 Generate Hero Images"):
    zip_data = process_images(hero_files, selected_hero_sizes, "hero")
    st.download_button("⬇️ Download All Hero Images", data=zip_data.getvalue(), file_name="heroes.zip", mime="application/zip")

if generated_images:
    st.markdown("### 🖼 Download Individual Files")
    for name, items in generated_images.items():
        st.subheader(name)
        for fname, img in items:
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG", optimize=True)
            st.image(img, caption=fname, width=200)
            st.download_button(f"⬇️ Download {fname}", data=img_bytes.getvalue(), file_name=fname, mime="image/png")
