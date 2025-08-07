
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import os
import zipfile
import tifffile

st.set_page_config(page_title="Athlete Image Generator", layout="centered")
st.title("🏋️ Athlete Avatar & Hero Image Generator")
st.write("Upload front and side profile images in TIFF, PNG, or JPG.")

front_files = st.file_uploader("Upload Front Profile Images", type=["tif", "tiff", "png", "jpg", "jpeg"], accept_multiple_files=True, key="front")
side_files = st.file_uploader("Upload Side Profile Images", type=["tif", "tiff", "png", "jpg", "jpeg"], accept_multiple_files=True, key="side")

avatar_sizes = [(256, 256), (500, 345)]
hero_sizes = [(1200, 1165), (1500, 920)]
selected_avatar_sizes = [s for s in avatar_sizes if st.checkbox(f"Avatar {s[0]}x{s[1]}", value=True)]
selected_hero_sizes = [s for s in hero_sizes if st.checkbox(f"Hero {s[0]}x{s[1]}", value=True)]

face_proto = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
face_model = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000_fp16.caffemodel"

@st.cache_resource
def load_dnn_model():
    try:
        proto_path = "deploy.prototxt"
        model_path = "res10_300x300_ssd_iter_140000.caffemodel"
        if not os.path.exists(proto_path):
            import urllib.request
            urllib.request.urlretrieve(face_proto, proto_path)
        if not os.path.exists(model_path):
            import urllib.request
            urllib.request.urlretrieve(face_model, model_path)
        return cv2.dnn.readNetFromCaffe(proto_path, model_path)
    except Exception as e:
        st.error(f"❌ Failed to load DNN model: {e}")
        return None

net = load_dnn_model()

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

def detect_face_dnn(image):
    try:
        array = np.array(image.convert("RGB"))
        h, w = array.shape[:2]
        blob = cv2.dnn.blobFromImage(array, 1.0, (300, 300), [104, 117, 123], False, False)
        net.setInput(blob)
        detections = net.forward()
        confidence_threshold = 0.5
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                h_margin = int(0.4 * (x2 - x1))
                v_margin = int(0.4 * (y2 - y1))
                return image.crop((max(x1 - h_margin, 0), max(y1 - v_margin, 0),
                                   min(x2 + h_margin, w), min(y2 + v_margin, h)))
    except Exception as e:
        st.error(f"❌ Face detection error: {e}")
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
            face = detect_face_dnn(image)
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
