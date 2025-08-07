
import streamlit as st
from PIL import Image
import os
import numpy as np
from mtcnn import MTCNN
import zipfile
import io
import tifffile

st.set_page_config(page_title="Athlete Image Generator", layout="centered")

st.title("🏋️ Athlete Avatar & Hero Image Generator")
st.write("Upload front and side profile images in TIFF, PNG, or JPG.")

# Uploaders
front_files = st.file_uploader("Upload Front Profile Images", type=["tif", "tiff", "png", "jpg", "jpeg"], accept_multiple_files=True, key="front")
side_files = st.file_uploader("Upload Side Profile Images", type=["tif", "tiff", "png", "jpg", "jpeg"], accept_multiple_files=True, key="side")

# Export options
st.subheader("Select Export Sizes")
avatar_sizes = [(256, 256), (500, 345)]
hero_sizes = [(1200, 1165), (1500, 920)]
selected_avatar_sizes = [s for s in avatar_sizes if st.checkbox(f"Avatar {s[0]}x{s[1]}", value=True)]
selected_hero_sizes = [s for s in hero_sizes if st.checkbox(f"Hero {s[0]}x{s[1]}", value=True)]

# Initialize face detector
detector = MTCNN()

def load_image(file):
    try:
        if file.name.lower().endswith((".tif", ".tiff")):
            image = tifffile.imread(file)
            image = Image.fromarray(image)
        else:
            image = Image.open(file)
        return image.convert("RGBA")
    except Exception as e:
        st.error(f"❌ Failed to load image: {file.name}, Error: {e}")
        return None

def detect_face(image):
    array = np.array(image)
    result = detector.detect_faces(array)
    if result:
        x, y, w, h = result[0]["box"]
        margin = int(0.4 * max(w, h))
        x1 = max(x - margin, 0)
        y1 = max(y - margin, 0)
        x2 = x + w + margin
        y2 = y + h + margin
        return image.crop((x1, y1, x2, y2))
    return None

def clean_filename(filename):
    name = os.path.splitext(filename)[0]
    if "-" in name:
        name = name.split("-")[0]
    return name

def process_and_export(files, sizes, label):
    output_zip = io.BytesIO()
    with zipfile.ZipFile(output_zip, "w") as zipf:
        for file in files:
            st.write(f"📂 Processing: {file.name}")
            image = load_image(file)
            if image is None:
                continue
            face = detect_face(image)
            if face is None:
                st.warning(f"⚠️ No face detected in {file.name}")
                continue
            for size in sizes:
                resized = face.resize(size, Image.LANCZOS)
                base = clean_filename(file.name)
                outname = f"{base}-{label}_{size[0]}x{size[1]}.png"
                buf = io.BytesIO()
                resized.save(buf, format="PNG", optimize=True)
                zipf.writestr(outname, buf.getvalue())
                st.success(f"✅ Exported: {outname}")
    output_zip.seek(0)
    return output_zip

if st.button("Generate Avatars"):
    if front_files and selected_avatar_sizes:
        zip_data = process_and_export(front_files, selected_avatar_sizes, "avatar")
        st.download_button("⬇️ Download Avatars ZIP", zip_data, file_name="avatars.zip")

if st.button("Generate Hero Images"):
    if side_files and selected_hero_sizes:
        zip_data = process_and_export(side_files, selected_hero_sizes, "hero")
        st.download_button("⬇️ Download Hero ZIP", zip_data, file_name="heroes.zip")
