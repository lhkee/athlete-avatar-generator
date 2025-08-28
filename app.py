import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io, os, zipfile
import sys, platform

# ───────────────────────── Crop safety knobs (fractions of OUTPUT height)
HAIR_MARGIN      = {"avatar": 0.022, "hero": 0.018}  # min space above detected hair top
BOTTOM_MARGIN    = {"avatar": 0.024, "hero": 0.020}  # min space below chin
HAIR_TALL_BONUS  = {"avatar": 0.010, "hero": 0.006}  # extra headroom for tall/afro hair

# ───────────────────────── Try Mediapipe (graceful)
try:
    import mediapipe as mp
    MP_OK = True
except Exception:
    MP_OK = False
    mp = None

st.set_page_config(page_title="Athlete Avatar & Hero Generator", layout="centered")
st.title("🏋️ Athlete Avatar & Hero Generator")
st.caption("Landmarks + hair-safe segmentation • Pillow TIFF (LZW) • Streamlit Free (Python 3.11).")

# ---------- Env panel (helps verify runtime on Streamlit Cloud)
def _env_panel():
    try:
        import mediapipe as _mp
        mp_ver = _mp.__version__
    except Exception:
        mp_ver = "NOT INSTALLED"
    st.sidebar.markdown("### Environment")
    st.sidebar.code(
        f"Python:   {sys.version.split()[0]}\n"
        f"Streamlit:{st.__version__}\n"
        f"NumPy:    {np.__version__}\n"
        f"OpenCV:   {cv2.__version__}\n"
        f"Mediapipe:{mp_ver}"
    )
_env_panel()

# ---------- UI toggles
cA, cB = st.columns(2)
with cA:
    debug = st.toggle("Debug Mode", value=False, help="Show overlays & crop info.")
with cB:
    auto_straighten = st.toggle("Auto-straighten (eyes)", value=True,
                                help="Deskew via eye line (Mediapipe when available).")

hair_safe = st.toggle(
    "Hair-safe mode (use segmentation if available)",
    value=True,
    help="Keeps full hairstyle using MediaPipe Selfie Segmentation; falls back to gradient scan."
)

# ---------- Uploaders
front_files = st.file_uploader("Front profile images (for Avatars)",
                               type=["tif","tiff","png","jpg","jpeg"], accept_multiple_files=True, key="front")
side_files  = st.file_uploader("Side profile images (for Heroes)",
                               type=["tif","tiff","png","jpg","jpeg"], accept_multiple_files=True, key="side")

# ---------- Export sizes and target lines (y / output_height)
st.subheader("Export sizes")
c1, c2 = st.columns(2)
with c1:
    avatar_opts = st.multiselect("Avatar", ["256x256","500x345"], default=["256x256","500x345"])
with c2:
    hero_opts   = st.multiselect("Hero",   ["1200x1165","1500x920"], default=["1200x1165","1500x920"])

TARGETS = {
    "avatar": {
        "256x256":  {"eye": 0.44, "chin": 0.832},
        "500x345":  {"eye": 0.44, "chin": 0.835},
    },
    "hero": {
        "1200x1165": {"eye": 0.35, "chin": 0.418},
        "1500x920":  {"eye": 0.35, "chin": 0.417},
    },
}

def _log(msg: str):
    if debug: st.write(msg)

# ---------- Loader (Pillow-first; supports TIFF LZW). Downscale huge images for stability.
def _load_image(upload):
    try:
        raw = upload.read()
        upload.seek(0)
        img = Image.open(io.BytesIO(raw))
        img.load()  # force decode
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        max_dim = 2600
        w, h = img.size
        if max(w, h) > max_dim:
            s = max_dim / float(max(w, h))
            nw, nh = int(w*s), int(h*s)
            _log(f"🔧 Downscaling {w}x{h} → {nw}x{nh}")
            img = img.resize((nw, nh), Image.LANCZOS)
        return img
    except Exception as e:
        st.error(f"❌ Failed to load {upload.name}: {e}")
        return None

# ---------- Haar cascades (always available)
FRONTAL = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
PROFILE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

def _detect_face_box(pil_img):
    bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = FRONTAL.detectMultiScale(gray, 1.1, 4, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(60,60))
    if len(faces) == 0:
        faces = PROFILE.detectMultiScale(gray, 1.1, 3, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(60,60))
    if len(faces) == 0:
        gray_flipped = cv2.flip(gray, 1)
        faces_flip = PROFILE.detectMultiScale(gray_flipped, 1.1, 3, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(60,60))
        if len(faces_flip) > 0:
            h, w = gray.shape
            faces = [(w-(x+wf), y, wf, hf) for (x, y, wf, hf) in faces_flip]
    if len(faces) == 0:
        return None
    x, y, w, h = [int(v) for v in sorted(faces, key=lambda b: b[2]*b[3], reverse=True)[0]]
    return (x, y, w, h)

# ---------- Mediapipe helpers (only if available)
if MP_OK:
    mp_face = mp.solutions.face_mesh
    mp_seg  = mp.solutions.selfie_segmentation
    LM_CHIN = 152
    LM_FOREHEAD = 10
    LM_EYES = [33, 133, 362, 263, 159, 386]  # corners + upper lids

    def _mediapipe_landmarks(pil_img):
        img = np.array(pil_img.convert("RGB"))
        with mp_face.FaceMesh(static_image_mode=True, refine_landmarks=True,
                              max_num_faces=1, min_detection_confidence=0.5) as fm:
            res = fm.process(img)
            if not res.multi_face_landmarks:
                return None
            lm = res.multi_face_landmarks[0].landmark
            H, W = img.shape[0], img.shape[1]
            pts = np.array([[p.x*W, p.y*H] for p in lm], dtype=np.float32)
            return pts

    def _eye_chin_forehead_y(landmarks):
        eye_y = float(np.mean([landmarks[i,1] for i in LM_EYES]))
        chin_y = float(landmarks[LM_CHIN,1])
        forehead_y = float(landmarks[LM_FOREHEAD,1])
        eye_x = float(np.mean([landmarks[i,0] for i in LM_EYES]))
        return eye_y, chin_y, forehead_y, eye_x

    # —— Segmentation-based hair-top (more aggressive for afros)
    def _hair_top_via_segmentation(pil_img, x_center, face_w, forehead_y, band_half_frac=0.40):
        try:
            img_rgb = np.array(pil_img.convert("RGB"))
            H, W = img_rgb.shape[:2]
            with mp_seg.SelfieSegmentation(model_selection=1) as seg:
                res = seg.process(img_rgb)
                if res.segmentation_mask is None:
                    return None
                mask = res.segmentation_mask
                fg = (mask > 0.30).astype(np.uint8)  # lower threshold to catch wispy hair

            band_half = int(max(6, band_half_frac * max(face_w, 1)))
            x0 = max(0, int(x_center) - band_half)
            x1 = min(W - 1, int(x_center) + band_half)
            band = fg[:, x0:x1]

            start = int(max(forehead_y - 0.55 * H, 0))
            stop  = int(max(forehead_y - 2, 0))
            if stop <= start:
                return None

            col_any = np.any(band[start:stop, :] > 0, axis=1)
            if not np.any(col_any):
                return None

            rows = np.flatnonzero(col_any)
            idx = int(np.percentile(rows, 1))  # very near top, still robust
            return float(start + idx)
        except Exception:
            return None

    # —— Wide-band gradient fallback
    def _scan_top_of_hair(pil_img, forehead_y, x_center, face_w, max_scan_frac=0.35):
        gray = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
        H, W = gray.shape
        band_half = int(0.30 * max(face_w, 1))
        x0, x1 = max(0, int(x_center - band_half)), min(W-1, int(x_center + band_half))

        start = int(max(forehead_y - max_scan_frac*H, 0))
        stop  = int(max(forehead_y - 2, 0))

        win = gray[:, x0:x1]
        grad = cv2.Sobel(win, cv2.CV_32F, 0, 1, ksize=3)
        gline = np.mean(np.abs(grad), axis=1)
        seg = gline[start:stop] if stop>start else gline[:1]

        if seg.size > 0:
            idx = int(np.argmax(seg))
            top_y = start + idx
        else:
            top_y = max(forehead_y - 0.08*H, 0)

        span = (forehead_y - top_y) / max(H,1)
        band = gray[max(0,int(top_y)):int(forehead_y), x0:x1]
        texture = float(np.var(band)) if band.size else 0.0
        if span > 0.18 or texture > 250.0:
            top_y = max(top_y - 0.04*H, 0)

        return float(top_y)

    def _face_shape_adjustments(eye_y, chin_y, forehead_y):
        face_h = max(chin_y - eye_y, 1.0)
        upper_h = max(eye_y - forehead_y, 1.0)
        ratio = face_h / upper_h
        adj = dict(eye_bias=0.0, scale_bias=1.0, shift_bias=0.0)
        if ratio < 1.5:      # wider faces: lift eye slightly (eyes end up a touch higher)
            adj["eye_bias"] = -0.012
            adj["shift_bias"] = -0.015
        elif ratio > 2.2:    # longer faces: enlarge & nudge down a bit
            adj["eye_bias"] = +0.015
            adj["scale_bias"] = 1.035
            adj["shift_bias"] = +0.012
        return adj

    def _deskew_by_eyes(pil_img, landmarks):
        pL = landmarks[33]; pR = landmarks[263]
        dy, dx = (pR[1]-pL[1]), (pR[0]-pL[0])
        if abs(dx) < 1e-3: return pil_img
        ang = np.degrees(np.arctan2(dy, dx))
        if abs(ang) < 1.5:  return pil_img
        if debug: st.write(f"↩️ Auto-straighten: {-ang:.2f}°")
        return pil_img.rotate(-ang, resample=Image.BICUBIC, expand=True)

    def _crop_from_landmarks(pil_img, face_box, landmarks, target_label, kind):
        tw, th = map(int, target_label.split("x"))
        ar = tw / float(th)
        (x,y,w,h) = face_box
        cx = x + w/2.0
        W_img, H_img = pil_img.size

        eye_y, chin_y, forehead_y, eye_x = _eye_chin_forehead_y(landmarks)

        # 1) segmentation hair-top (if enabled), 2) gradient fallback
        top_y = None
        if hair_safe:
            top_y = _hair_top_via_segmentation(pil_img, x_center=cx, face_w=w, forehead_y=forehead_y)
        if top_y is None:
            top_y = _scan_top_of_hair(pil_img, forehead_y, x_center=cx, face_w=w)

        # keep the same eye–chin geometry you signed off on
        adj = _face_shape_adjustments(eye_y, chin_y, forehead_y)
        tgt = TARGETS[kind][target_label].copy()
        tgt["eye"] = np.clip(tgt["eye"] + adj["eye_bias"], 0.05, 0.95)

        crop_h = (chin_y - eye_y) / max((tgt["chin"] - tgt["eye"]), 1e-6)
        crop_h *= adj["scale_bias"]

        # initial top from eye target
        crop_top = eye_y - tgt["eye"] * crop_h
        crop_top += adj["shift_bias"] * th

        # ---- Hair-top safety (upper) ----
        hair_span = (forehead_y - top_y) / max(H_img, 1.0)
        top_margin = HAIR_MARGIN[kind] * th
        if hair_span > 0.20:                         # tall/afro hair → extra headroom
            top_margin += HAIR_TALL_BONUS[kind] * th
        if (top_y - crop_top) < top_margin:
            crop_top = max(0.0, top_y - top_margin)

        # ---- Chin-bottom safety (lower) ----
        crop_bottom = crop_top + crop_h
        bottom_margin = BOTTOM_MARGIN[kind] * th
        delta = (chin_y + bottom_margin) - crop_bottom
        if delta > 0:
            crop_top += delta
            crop_bottom += delta

        # If both constraints fight, gently expand height to satisfy both
        if crop_top < 0:
            expand = -crop_top
            crop_h += expand
            crop_top = 0
            crop_bottom = crop_top + crop_h

        # Width & sides
        crop_w = crop_h * ar
        crop_left = cx - crop_w/2.0
        crop_right = crop_left + crop_w

        # clamp to image bounds
        if crop_left < 0:
            crop_right -= crop_left; crop_left = 0
        if crop_right > W_img:
            shift = crop_right - W_img
            crop_left -= shift; crop_right = W_img
        if crop_bottom > H_img:
            shift = crop_bottom - H_img
            crop_top -= shift; crop_bottom = H_img
        if crop_top < 0:
            crop_bottom -= crop_top; crop_top = 0

        L = int(max(0, round(crop_left)))
        T = int(max(0, round(crop_top)))
        R = int(min(W_img, round(crop_right)))
        B = int(min(H_img, round(crop_bottom)))
        if R <= L+10 or B <= T+10:
            # extreme fallback: expand around face box by aspect
            ar2 = tw/float(th)
            x0,y0,fw,fh = (x,y,w,h)
            ch2 = int(max(fh*1.6, 60))
            cw2 = int(ch2*ar2)
            cx0 = x0 + fw/2; cy0 = y0 + fh/2
            L = int(max(0, cx0 - cw2/2)); T = int(max(0, cy0 - ch2/2))
            R = int(min(W_img, L+cw2));    B = int(min(H_img, T+ch2))

        crop = pil_img.crop((L,T,R,B))
        dbg = {"crop_top": float(T), "crop_bottom": float(B), "top_y": float(top_y)}
        return crop, dbg

# ---------- Simple face-centered fallback (Haar only)
def _face_center_crop(pil_img, face_box, target_wh, kind):
    tw, th = target_wh
    ar = tw / float(th)
    (x,y,w,h) = face_box
    cx, cy = x + w/2.0, y + h/2.0
    face_frac = 0.60 if kind=="avatar" else 0.52
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
    return pil_img.crop((left, top, right, bottom)), None

# ---------- Naming rule: use original name up to first '-' (drop trailing numbers), add suffix
def _clean_base(filename):
    base = os.path.splitext(filename)[0]
    if "-" in base:
        base = base.split("-")[0]
    return base

# ---------- Pipeline (batch → ZIP download)
def _process(files, sizes, label):
    if not files or not sizes:
        st.warning("⚠️ No files or sizes selected."); return

    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for up in files:
            st.write(f"📂 **{up.name}**")
            img = _load_image(up)
            if img is None: continue

            face = _detect_face_box(img)
            if face is None:
                st.warning(f"⚠️ No face detected in {up.name}. Skipped."); continue

            # Landmarks if available
            lms = None
            if MP_OK:
                try:
                    lms = _mediapipe_landmarks(img)
                except Exception as e:
                    lms = None
                    if debug: st.write(f"⚠️ Mediapipe landmark error: {e}")

            # Deskew if possible
            if auto_straighten and lms is not None:
                img = _deskew_by_eyes(img, lms)
                try:
                    lms = _mediapipe_landmarks(img)
                except Exception:
                    lms = None

            base = _clean_base(up.name)

            for s in sizes:
                kind = "avatar" if label=="avatar" else "hero"
                w, h = map(int, s.split("x"))

                if lms is not None and MP_OK:
                    crop, dbg = _crop_from_landmarks(img, face, lms, s, kind)
                else:
                    crop, dbg = _face_center_crop(img, face, (w,h), kind)

                out = crop.resize((w,h), Image.LANCZOS)

                if debug:
                    pr = np.array(out)
                    tgt = TARGETS.get(kind, {}).get(s, None)
                    if tgt is not None:
                        ey = int(tgt["eye"]  * h)
                        cy = int(tgt["chin"] * h)
                        cv2.line(pr, (0, ey), (w-1, ey), (0,0,255), 1)   # eye target (red)
                        cv2.line(pr, (0, cy), (w-1, cy), (0,0,255), 1)   # chin target (red)
                    if dbg is not None:
                        try:
                            crop_top = dbg["crop_top"]; crop_bottom = dbg["crop_bottom"]; top_y = dbg["top_y"]
                            scale_y = h / float((crop_bottom - crop_top) or 1.0)
                            out_top = int((top_y - crop_top) * scale_y)
                            out_top = np.clip(out_top, 0, h-1)
                            cv2.line(pr, (0, out_top), (w-1, out_top), (0,255,255), 1)  # hair top (cyan)
                        except Exception:
                            pass
                    st.image(pr, caption=f"Preview ({s})", use_column_width=True)

                buf = io.BytesIO()
                out.save(buf, format="PNG", optimize=True)
                out_name = f"{base}-{label}_{w}x{h}.png"
                zf.writestr(out_name, buf.getvalue())
                if debug: st.success(f"✅ Exported: {out_name}")

    mem_zip.seek(0)
    st.download_button(f"⬇️ Download {label.title()} ZIP", mem_zip,
                       file_name=f"{label}_images.zip", mime="application/zip")

# ---------- Actions
cX, cY = st.columns(2)
with cX:
    if st.button("✅ Generate Avatars"):
        _process(front_files, avatar_opts, "avatar")
with cY:
    if st.button("✅ Generate Hero Images"):
        _process(side_files, hero_opts, "hero")
