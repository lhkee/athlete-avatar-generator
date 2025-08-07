import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Athlete Image Generator", layout="centered")

st.title("🏋️ Athlete Image Generator")
st.markdown("Upload front (avatar) or side (hero) profile images below. The app will export transparent PNGs in selected sizes.")

avatar_images = st.file_uploader("📤 Upload Front Profile Image(s) for Avatar", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True)
hero_images = st.file_uploader("📤 Upload Side Profile Image(s) for Hero", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True)

if avatar_images:
    st.success(f"✅ {len(avatar_images)} avatar image(s) uploaded.")
if hero_images:
    st.success(f"✅ {len(hero_images)} hero image(s) uploaded.")

st.markdown("### 🛠 Export Sizes (Default checked)")
avatar_sizes = st.multiselect("Avatar Sizes", options=["256x256", "500x345"], default=["256x256", "500x345"])
hero_sizes = st.multiselect("Hero Sizes", options=["1200x1165", "1500x920"], default=["1200x1165", "1500x920"])

def resize_and_export(image, size_str):
    w, h = map(int, size_str.split("x"))
    resized = image.resize((w, h), Image.ANTIALIAS)
    output = io.BytesIO()
    resized.save(output, format="PNG", optimize=True)
    return output.getvalue()

def process_images(images, sizes, label):
    for img_file in images:
        try:
            base = img_file.name.rsplit(".", 1)[0]
            name = base.split("-")[0]
            pil_image = Image.open(img_file).convert("RGBA")
            for size in sizes:
                result = resize_and_export(pil_image, size)
                filename = f"{name}-{label}_{size}.png"
                st.download_button(f"⬇️ Download {filename}", data=result, file_name=filename, mime="image/png")
                st.success(f"✅ Generated {filename}")
        except Exception as e:
            st.error(f"❌ Error processing {img_file.name}: {str(e)}")

if avatar_images and avatar_sizes:
    if st.button("🎨 Generate Avatars"):
        st.markdown("### 🖼 Avatar Exports")
        process_images(avatar_images, avatar_sizes, "avatar")

if hero_images and hero_sizes:
    if st.button("🎨 Generate Hero Images"):
        st.markdown("### 🖼 Hero Exports")
        process_images(hero_images, hero_sizes, "hero")