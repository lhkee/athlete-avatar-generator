import os
import io
import zipfile
import streamlit as st
import numpy as np
from PIL import Image
import cv2

st.set_page_config(page_title="Athlete Image Generator", layout="centered")
st.title("🏋️ Athlete Image Generator")

avatar_files = st.file_uploader("📤 Upload Front Profile Images", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True, key="front")
hero_files = st.file_uploader("📤 Upload Side Profile Images", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True, key="side")

avatar_sizes = ["256x256", "500x345"]
hero_sizes = ["1200x1165", "1500x920"]

selected_avatar_sizes = st.multiselect("Avatar Export Sizes", avatar_sizes, default=avatar_sizes)
selected_hero_sizes = st.multiselect("Hero Export Sizes", hero_sizes, default=hero_sizes)

generated_images = {}

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def detect_and_crop_face(pil_img):
    try:
        img_rgba = np.array(pil_img)
        h, w = img_rgba.shape[:2]
        img_rgb = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2RGB)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=3)
        if len(faces) == 0:
            st.warning("⚠️ No face detected, using original image")
            return pil_img  # fallback

        x, y, fw, fh = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)[0]
        buffer = int(0.6 * fh)
        left = max(0, x - buffer)
        top = max(0, y - buffer)
        right = min(img_rgb.shape[1], x + fw + buffer)
        bottom = min(img_rgb.shape[0], y + fh + buffer)

        cropped = img_rgb[top:bottom, left:right]
        return Image.fromarray(cropped).convert("RGBA")
    except Exception as e:
        st.error(f"Face crop error: {e}")
        return pil_img

def place_on_canvas(cropped_img, size_str):
    try:
        target_w, target_h = map(int, size_str.split("x"))
        canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
        aspect_ratio = cropped_img.width / cropped_img.height
        if target_w / target_h > aspect_ratio:
            new_h = target_h
            new_w = int(aspect_ratio * new_h)
        else:
            new_w = target_w
            new_h = int(new_w / aspect_ratio)
        resized = cropped_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        x = (target_w - new_w) // 2
        y = (target_h - new_h) // 2
        canvas.paste(resized, (x, y), resized)
        return canvas
    except Exception as e:
        st.error(f"Canvas error: {e}")
        return cropped_img

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
                    export = place_on_canvas(cropped, size)
                    filename = f"{name}-{label}_{size}.png"
                    if name not in generated_images:
                        generated_images[name] = []
                    generated_images[name].append((filename, export))
                    add_to_zip(zipf, filename, export)
                st.success(f"✅ Processed: {file.name}")
            except Exception as e:
                st.error(f"❌ Error with {file.name}: {e}")
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