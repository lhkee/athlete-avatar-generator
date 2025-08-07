
import streamlit as st
from PIL import Image
import io
import os
import numpy as np
import zipfile
from facenet_pytorch import MTCNN
import tempfile
from datetime import datetime

st.set_page_config(page_title="Athlete Image Generator", layout="centered")
st.title("🏋️‍♂️ Athlete Image Generator")
st.write("Upload front (avatar) or side (hero) profile images below. The app will export transparent PNGs in selected sizes.")

# Load MTCNN
mtcnn = MTCNN(keep_all=False, device='cpu')

# File upload widgets
front_files = st.file_uploader("📤 Upload Front Profile Image(s) for Avatar", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True)
side_files = st.file_uploader("📤 Upload Side Profile Image(s) for Hero", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True)

# Size selectors
st.subheader("🛠 Export Sizes (Default checked)")
avatar_sizes = st.multiselect("Avatar Sizes", options=["256x256", "500x345"], default=["256x256", "500x345"])
hero_sizes = st.multiselect("Hero Sizes", options=["1200x1165", "1500x920"], default=["1200x1165", "1500x920"])

# Utility function to crop and resize
def generate_face_crop(img, sizes, suffix):
    results = []
    boxes, _ = mtcnn.detect(img)
    if boxes is None:
        return []
    box = boxes[0]  # single face assumed
    x1, y1, x2, y2 = map(int, box)
    face_crop = img.crop((x1, y1, x2, y2))
    for size in sizes:
        w, h = map(int, size.split('x'))
        resized = face_crop.resize((w, h), Image.LANCZOS)
        results.append((resized, f"{suffix}_{size}.png"))
    return results

# TIFF conversion function
def convert_tif_to_rgb(image_file):
    try:
        img = Image.open(image_file)
        return img.convert("RGB")
    except:
        return None

# Generation and download logic
if st.button("✅ Generate Images"):
    all_outputs = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for files, category, sizes in [(front_files, "avatar", avatar_sizes), (side_files, "hero", hero_sizes)]:
            for file in files:
                filename = os.path.splitext(file.name)[0]
                base_filename = filename.split('-')[0]  # strip suffix
                image = convert_tif_to_rgb(file)
                if image is None:
                    st.error(f"❌ Could not read file: {file.name}")
                    continue
                results = generate_face_crop(image, sizes, f"{base_filename}-{category}")
                for img, name in results:
                    save_path = os.path.join(temp_dir, name)
                    img.save(save_path, format="PNG")
                    all_outputs.append((name, save_path))
        if all_outputs:
            zip_path = os.path.join(temp_dir, "athlete_images.zip")
            with zipfile.ZipFile(zip_path, "w") as zipf:
                for name, path in all_outputs:
                    zipf.write(path, arcname=name)
            with open(zip_path, "rb") as f:
                st.download_button("📦 Download All as ZIP", f, file_name="athlete_images.zip")
        else:
            st.warning("⚠️ No images were processed.")
