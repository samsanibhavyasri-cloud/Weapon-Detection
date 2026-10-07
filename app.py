import streamlit as st
import cv2
import os
import json
import math
import tempfile
import subprocess
import time
import urllib.parse
import urllib.request
import base64
import threading
from datetime import datetime

# Browser webcam support for Streamlit Cloud / remote deployment.
try:
    import av
    from streamlit_webrtc import WebRtcMode, webrtc_streamer
    WEBRTC_AVAILABLE = True
except Exception:
    av = None
    WebRtcMode = None
    webrtc_streamer = None
    WEBRTC_AVAILABLE = False

import numpy as np
from PIL import Image
from ultralytics import YOLO


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="WeaponGuard AI",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT SETTINGS
# ============================================================

MODEL_PATH = "best.pt"
ALERT_SOUND = "alert.mp3"
LOG_FILE = "detection_log.json"

COLLEGE_NAME = "VSM College of Engineering"
GUIDE_NAME = "Mr. Abdul Aziz MD"

TEAM_LEAD = "S. Nagasindhu"
TEAM_MEMBERS = [
    "S. Bhavyasri",
    "S. Manasa",
    "S. Anusha"
]

DEFAULT_CONFIDENCE = 0.60
DEFAULT_REQUIRED_FRAMES = 5

# Very small boxes are ignored. This helps reduce
# random tiny false detections.
MIN_BOX_AREA_RATIO = 0.0015

# Phone/SMS alert settings. Keep credentials in Windows environment
# variables instead of writing them directly in app.py.
SMS_ENABLED = os.getenv("WEAPONGUARD_SMS_ENABLED", "false").lower() == "true"
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "")
ALERT_TO_NUMBER = os.getenv("WEAPONGUARD_ALERT_TO", "")
SMS_COOLDOWN_SECONDS = 60

if "last_sms_alert_time" not in st.session_state:
    st.session_state.last_sms_alert_time = 0.0


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

if "theme" not in st.session_state:
    st.session_state.theme = "System"


# ============================================================
# THEME + CSS
# ============================================================

def apply_theme():

    theme = st.session_state.theme

    if theme == "Light":
        bg = "#ffffff"
        card = "#f7f8fa"
        text = "#17191f"
        secondary = "#5f6368"
        border = "#d9dce1"
        sidebar = "#f3f4f6"

    elif theme == "Dark":
        bg = "#0e1117"
        card = "#181c25"
        text = "#f4f5f7"
        secondary = "#aeb4bf"
        border = "#30343d"
        sidebar = "#151923"

    else:
        # System mode: CSS media query is used below.
        bg = "#ffffff"
        card = "#ffffff"
        text = "#17191f"
        secondary = "#5f6368"
        border = "#d9dce1"
        sidebar = "#f5f6f8"

    st.markdown(
        f"""
        <style>
        :root {{
            --wg-bg: {bg};
            --wg-card: {card};
            --wg-text: {text};
            --wg-secondary: {secondary};
            --wg-border: {border};
            --wg-sidebar: {sidebar};
            --wg-red: #e53935;
            --wg-green: #16a34a;
        }}

        @media (prefers-color-scheme: dark) {{
            :root {{
                --wg-bg: #0e1117;
                --wg-card: #181c25;
                --wg-text: #f4f5f7;
                --wg-secondary: #aeb4bf;
                --wg-border: #30343d;
                --wg-sidebar: #151923;
            }}
        }}

        {"@media (prefers-color-scheme: light) { :root { --wg-bg:#ffffff; --wg-card:#ffffff; --wg-text:#17191f; --wg-secondary:#5f6368; --wg-border:#d9dce1; --wg-sidebar:#f5f6f8; } }" if theme == "System" else ""}

        .stApp {{
            background: var(--wg-bg);
            color: var(--wg-text);
        }}

        [data-testid="stSidebar"] {{
            background: var(--wg-sidebar);
        }}

        [data-testid="stSidebar"] * {{
            color: var(--wg-text);
        }}

        h1, h2, h3, h4, h5, h6,
        p, label, span {{
            color: var(--wg-text);
        }}

        .wg-title {{
            text-align: center;
            font-size: 48px;
            font-weight: 800;
            color: var(--wg-text);
            margin-top: 20px;
            margin-bottom: 5px;
        }}

        .wg-subtitle {{
            text-align: center;
            color: var(--wg-secondary);
            font-size: 18px;
            margin-bottom: 30px;
        }}

        .wg-card {{
            background: var(--wg-card);
            border: 1px solid var(--wg-border);
            border-radius: 16px;
            padding: 25px;
        }}

        .wg-description {{
            max-width: 950px;
            margin: auto;
            text-align: center;
            color: var(--wg-text);
            font-size: 18px;
            line-height: 1.8;
        }}

        .wg-danger {{
            background: #7f1d1d;
            color: white !important;
            border-radius: 12px;
            padding: 18px;
            text-align: center;
            font-size: 20px;
            font-weight: 700;
        }}

        .wg-safe {{
            background: #14532d;
            color: #dcfce7 !important;
            border-radius: 12px;
            padding: 18px;
            text-align: center;
            font-size: 20px;
            font-weight: 700;
        }}

        .wg-sidebar-title {{
            font-size: 24px;
            font-weight: 800;
            line-height: 1.35;
            text-align: center;
        }}

        .wg-small {{
            color: var(--wg-secondary);
            font-size: 14px;
        }}

        .stButton > button {{
            border-radius: 9px;
            font-weight: 650;
            min-height: 42px;
        }}

        div.stButton > button[kind="primary"] {{
            background: #e53935;
            border-color: #e53935;
            color: white !important;
        }}

        div.stButton > button[kind="primary"]:hover {{
            background: #c62828;
            border-color: #c62828;
            color: white !important;
        }}

        [data-testid="stFileUploader"] {{
            border: 1px solid var(--wg-border);
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


apply_theme()


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    st.error("❌ best.pt was not found in the project folder.")
    st.stop()

model = load_model()


# ============================================================
# LOGGING
# ============================================================

def save_detection_log(source, confidence, count=1):

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "confidence": round(float(confidence), 3),
        "weapon_count": int(count)
    }

    logs = []

    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                old = json.load(f)

            if isinstance(old, list):
                logs = old
            elif isinstance(old, dict):
                # Supports the older {"detections": [...]} format
                if isinstance(old.get("detections"), list):
                    logs = old["detections"]
                else:
                    logs = [old]

    except Exception:
        logs = []

    logs.append(record)
    logs = logs[-100:]

    try:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=4)
    except Exception:
        pass


# ============================================================
# PHONE / SMS ALERT
# ============================================================

def send_phone_alert(message, force=False):
    """Send an SMS using Twilio when a confirmed weapon event occurs."""

    if not SMS_ENABLED:
        return False, "SMS alerts are disabled."

    missing = []
    if not TWILIO_ACCOUNT_SID:
        missing.append("TWILIO_ACCOUNT_SID")
    if not TWILIO_AUTH_TOKEN:
        missing.append("TWILIO_AUTH_TOKEN")
    if not TWILIO_FROM_NUMBER:
        missing.append("TWILIO_FROM_NUMBER")
    if not ALERT_TO_NUMBER:
        missing.append("WEAPONGUARD_ALERT_TO")

    if missing:
        return False, "Missing environment variables: " + ", ".join(missing)

    now = time.time()
    if not force and now - st.session_state.last_sms_alert_time < SMS_COOLDOWN_SECONDS:
        return False, "SMS cooldown is active."

    url = (
        "https://api.twilio.com/2010-04-01/Accounts/"
        f"{TWILIO_ACCOUNT_SID}/Messages.json"
    )

    payload = urllib.parse.urlencode({
        "From": TWILIO_FROM_NUMBER,
        "To": ALERT_TO_NUMBER,
        "Body": message[:1500]
    }).encode("utf-8")

    credentials = f"{TWILIO_ACCOUNT_SID}:{TWILIO_AUTH_TOKEN}".encode("utf-8")
    auth = base64.b64encode(credentials).decode("ascii")

    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/x-www-form-urlencoded"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            status = response.status
            if 200 <= status < 300:
                st.session_state.last_sms_alert_time = now
                return True, "SMS alert sent successfully."
            return False, f"Twilio returned HTTP {status}."
    except Exception as e:
        return False, f"SMS sending failed: {e}"


def build_alert_message(source, confidence, event_time=None):
    time_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    video_time = "" if event_time is None else f"\n🎥 Video time: {event_time:.2f} sec"
    return (
        "🚨 WEAPONGUARD AI ALERT\n\n"
        "A weapon was detected by the CCTV AI system.\n"
        f"📍 Location: {COLLEGE_NAME}\n"
        f"🕒 Alert time: {time_text}\n"
        f"📷 Source: {source}\n"
        f"🎯 Confidence: {confidence:.1%}"
        f"{video_time}\n\n"
        "Please check the CCTV feed and verify the event."
    )


# ============================================================
# DETECT WEAPONS IN ONE FRAME
# ============================================================

def detect_weapons(frame, confidence_threshold):

    output = frame.copy()
    detections = []

    h, w = frame.shape[:2]
    frame_area = max(1, h * w)

    results = model(
        frame,
        conf=confidence_threshold,
        verbose=False
    )

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = str(model.names[class_id]).lower().strip()

            if class_name != "weapon":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            box_area = max(0, x2 - x1) * max(0, y2 - y1)
            area_ratio = box_area / frame_area

            # Ignore extremely tiny detections.
            if area_ratio < MIN_BOX_AREA_RATIO:
                continue

            detections.append({
                "box": (x1, y1, x2, y2),
                "confidence": confidence
            })

    # Draw every accepted weapon detection.
    for det in detections:

        x1, y1, x2, y2 = det["box"]
        confidence = det["confidence"]

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3
        )

        cv2.putText(
            output,
            f"WEAPON {confidence:.2f}",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    highest = max(
        [d["confidence"] for d in detections],
        default=0.0
    )

    return output, detections, highest


# ============================================================
# BANNER
# ============================================================

def add_banner(frame, text, color):

    cv2.rectangle(
        frame,
        (0, 0),
        (frame.shape[1], 65),
        color,
        -1
    )

    cv2.putText(
        frame,
        text,
        (20, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        3
    )

    return frame


# ============================================================
# FFMPEG CHECK
# ============================================================

def ffmpeg_available():

    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


# ============================================================
# CREATE BROWSER-COMPATIBLE VIDEO
# ============================================================

def encode_browser_video(input_path, output_path):

    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,

        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",

        "-c:a", "aac",
        "-b:a", "128k",

        "-movflags", "+faststart",

        output_path
    ]

    try:

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=600
        )

        if (
            result.returncode == 0
            and os.path.exists(output_path)
            and os.path.getsize(output_path) > 0
        ):
            return True, ""

        return False, result.stderr[-4000:]

    except Exception as e:
        return False, str(e)


# ============================================================
# CREATE ALARM AUDIO
# ============================================================

def create_alarm_video(
    silent_video,
    final_video,
    events,
    original_video=None
):
    """
    Creates an H.264/AAC MP4 and places a short alert.mp3
    at every confirmed detection event.

    Original CCTV audio is also retained when the source has
    an audio stream.
    """

    if not os.path.exists(ALERT_SOUND):
        return False, "alert.mp3 was not found."

    if not events:
        return False, "No detection events were supplied."

    # Keep events at least 2 seconds apart.
    clean_events = []
    last_time = -999

    for event in events:

        try:
            event_time = float(event["time"])
        except Exception:
            continue

        if event_time - last_time >= 2.0:
            clean_events.append(event_time)
            last_time = event_time

    if not clean_events:
        return False, "No valid alarm events."

    # We use the processed video as the video stream.
    # It has no audio, so the alarm can be mixed independently.
    inputs = ["-i", silent_video]
    filter_parts = []
    labels = []

    for i, event_time in enumerate(clean_events):

        input_index = i + 1
        delay_ms = max(0, int(event_time * 1000))
        label = f"a{i}"

        inputs.extend([
            "-stream_loop", "-1",
            "-i", ALERT_SOUND
        ])

        filter_parts.append(
            f"[{input_index}:a]"
            f"atrim=duration=2,"
            f"asetpts=N/SR/TB,"
            f"adelay={delay_ms}|{delay_ms}"
            f"[{label}]"
        )

        labels.append(f"[{label}]")

    filter_complex = ";".join(filter_parts)

    filter_complex += (
        ";"
        + "".join(labels)
        + f"amix=inputs={len(labels)}:"
          "duration=longest:"
          "dropout_transition=0,"
          "aresample=async=1"
          "[alarm]"
    )

    command = [
        "ffmpeg",
        "-y"
    ]

    command.extend(inputs)

    command.extend([
        "-filter_complex",
        filter_complex,

        "-map", "0:v:0",
        "-map", "[alarm]",

        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",

        "-c:a", "aac",
        "-b:a", "128k",

        "-movflags", "+faststart",
        "-shortest",

        final_video
    ])

    try:

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=600
        )

        if (
            result.returncode == 0
            and os.path.exists(final_video)
            and os.path.getsize(final_video) > 0
        ):
            return True, ""

        return False, result.stderr[-5000:]

    except Exception as e:
        return False, str(e)


# ============================================================
# VIDEO PROCESSING
# ============================================================

def process_video(
    input_path,
    silent_output,
    confidence_threshold,
    required_frames
):

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        return {
            "success": False,
            "error": "Could not open the uploaded video."
        }

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        cap.release()
        return {
            "success": False,
            "error": "Invalid video dimensions."
        }

    # MP4 written temporarily. FFmpeg later converts it to
    # proper browser-compatible H.264.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        silent_output,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        return {
            "success": False,
            "error": "Could not create output video."
        }

    frame_number = 0
    consecutive_frames = 0
    was_confirmed = False

    confirmed_frames = 0
    highest_confidence = 0.0

    detection_events = []

    progress = st.progress(0)
    status = st.empty()
    preview = st.empty()

    try:

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            output, detections, highest = detect_weapons(
                frame,
                confidence_threshold
            )

            if detections:

                consecutive_frames += 1
                highest_confidence = max(
                    highest_confidence,
                    highest
                )

            else:

                consecutive_frames = 0

            confirmed = (
                consecutive_frames >= required_frames
            )

            # ------------------------------------------------
            # NEW CONFIRMED DETECTION EVENT
            # ------------------------------------------------

            if confirmed and not was_confirmed:

                time_sec = frame_number / fps

                detection_events.append({
                    "time": round(time_sec, 2),
                    "confidence": round(highest, 3)
                })

                confirmed_frames += 1

                save_detection_log(
                    "CCTV Video",
                    highest,
                    len(detections)
                )

            was_confirmed = confirmed

            # ------------------------------------------------
            # DISPLAY BANNER
            # ------------------------------------------------

            if confirmed:

                output = add_banner(
                    output,
                    "🚨 WEAPON DETECTED",
                    (0, 0, 180)
                )

                status.error(
                    f"🚨 WEAPON DETECTED | "
                    f"Confidence: {highest:.2%}"
                )

            elif detections:

                output = add_banner(
                    output,
                    "VERIFYING POSSIBLE WEAPON...",
                    (0, 120, 220)
                )

                status.warning(
                    "🔎 Possible weapon detected - verifying..."
                )

            else:

                output = add_banner(
                    output,
                    "NO WEAPON DETECTED",
                    (0, 80, 0)
                )

                status.success(
                    "✅ No confirmed weapon in this frame"
                )

            writer.write(output)

            # ------------------------------------------------
            # PREVIEW
            # ------------------------------------------------

            preview_rgb = cv2.cvtColor(
                output,
                cv2.COLOR_BGR2RGB
            )

            preview.image(
                preview_rgb,
                channels="RGB",
                use_container_width=True
            )

            if total_frames > 0:

                value = min(
                    frame_number / total_frames,
                    1.0
                )

                progress.progress(value)

    finally:

        cap.release()
        writer.release()

    progress.progress(1.0)
    preview.empty()

    return {
        "success": True,
        "fps": fps,
        "width": width,
        "height": height,
        "frames": frame_number,
        "events": detection_events,
        "confirmed_detections": len(detection_events),
        "highest_confidence": highest_confidence
    }


# ============================================================
# AUDIO PREVIEW
# ============================================================

def show_alarm_player():

    if os.path.exists(ALERT_SOUND):

        with open(ALERT_SOUND, "rb") as f:
            alarm_bytes = f.read()

        st.audio(
            alarm_bytes,
            format="audio/mp3"
        )


# ============================================================
# HOME PAGE
# ============================================================

def home_page():

    with st.sidebar:

        st.markdown(
            """
            <div class="wg-sidebar-title">
                Artificial Intelligence<br>
                Career for Women (AICW)
            </div>
            """,
            unsafe_allow_html=True
        )

        st.divider()

        st.markdown("### 🎓 College")
        st.write(COLLEGE_NAME)

        st.divider()

        st.markdown("### 👥 Team Members")

        st.markdown(
            f"**⭐ {TEAM_LEAD} — Team Lead**"
        )

        for member in TEAM_MEMBERS:
            st.write(f"• {member} — Team Member")

        st.divider()

        st.markdown("### 👨‍🏫 Project Guide")
        st.write(GUIDE_NAME)

        st.divider()

        st.markdown("### 🎨 Appearance")

        theme = st.selectbox(
            "Theme",
            ["System", "Light", "Dark"],
            index=["System", "Light", "Dark"].index(
                st.session_state.theme
            ),
            label_visibility="collapsed"
        )

        if theme != st.session_state.theme:
            st.session_state.theme = theme
            st.rerun()

    st.markdown(
        '<div class="wg-title">🚨 WeaponGuard AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="wg-subtitle">'
        'AI-Based Weapon Detection Using CCTV'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="wg-card">
            <h2 style="text-align:center;">Description</h2>
            <div class="wg-description">
                WeaponGuard AI is an intelligent weapon detection
                system designed to improve security through
                automated image, CCTV video, and live camera
                analysis. The system uses YOLO-based object
                detection to identify weapons and provide an
                alert when a weapon is confirmed.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    st.write("")

    if st.button(
        "➡️ NEXT",
        type="primary",
        use_container_width=True
    ):

        st.session_state.page = "detection"
        st.rerun()


# ============================================================
# IMAGE DETECTION
# ============================================================

def image_detection(confidence_threshold):

    st.subheader("📷 Image Detection")

    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="image_upload"
    )

    if uploaded is None:
        return

    image = Image.open(uploaded).convert("RGB")
    image_array = np.array(image)

    output, detections, highest = detect_weapons(
        cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2BGR
        ),
        confidence_threshold
    )

    output_rgb = cv2.cvtColor(
        output,
        cv2.COLOR_BGR2RGB
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("### 📥 Input Image")
        st.image(
            image,
            use_container_width=True
        )

    with col2:
        st.markdown("### 📤 Output Image")
        st.image(
            output_rgb,
            use_container_width=True
        )

    if detections:

        st.markdown(
            f"""
            <div class="wg-danger">
                🚨 WEAPON DETECTED<br>
                Count: {len(detections)}<br>
                Confidence: {highest:.2%}
            </div>
            """,
            unsafe_allow_html=True
        )

        save_detection_log(
            "Image Detection",
            highest,
            len(detections)
        )

        st.markdown("### 🔊 Alert Sound")
        show_alarm_player()

    else:

        st.markdown(
            """
            <div class="wg-safe">
                ✅ NO WEAPON DETECTED
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# VIDEO DETECTION
# ============================================================

def video_detection(
    confidence_threshold,
    required_frames
):

    st.subheader("🎥 CCTV Video Detection")

    uploaded = st.file_uploader(
        "Choose CCTV video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload"
    )

    if uploaded is None:
        return

    # --------------------------------------------------------
    # SAVE INPUT
    # --------------------------------------------------------

    input_temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    input_temp.write(
        uploaded.getbuffer()
    )

    input_temp.close()

    input_path = input_temp.name

    # --------------------------------------------------------
    # SHOW INPUT BEFORE PROCESSING
    # --------------------------------------------------------

    st.markdown("### 📥 Input CCTV Video")
    st.video(
        uploaded.getvalue()
    )

    st.write("")

    if not st.button(
        "🚀 START WEAPON DETECTION",
        type="primary",
        use_container_width=True,
        key="start_video_detection"
    ):
        return

    # --------------------------------------------------------
    # CHECK FFMPEG
    # --------------------------------------------------------

    if not ffmpeg_available():

        st.error(
            """
            ❌ FFmpeg is not available.

            Install FFmpeg and add it to Windows PATH.
            Then restart PowerShell and run the app again.
            """
        )

        return

    # --------------------------------------------------------
    # TEMP OUTPUTS
    # --------------------------------------------------------

    silent_temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix="_silent.mp4"
    )

    silent_temp.close()

    silent_path = silent_temp.name

    final_temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix="_alarm.mp4"
    )

    final_temp.close()

    final_path = final_temp.name

    browser_temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix="_browser.mp4"
    )

    browser_temp.close()

    browser_path = browser_temp.name

    # Remove placeholder files so FFmpeg can create them.
    for path in [
        silent_path,
        final_path,
        browser_path
    ]:
        try:
            os.remove(path)
        except Exception:
            pass

    # --------------------------------------------------------
    # PROCESS
    # --------------------------------------------------------

    st.markdown("### 🔄 Processing video...")

    result = process_video(
        input_path,
        silent_path,
        confidence_threshold,
        required_frames
    )

    if not result.get("success"):

        st.error(
            "❌ " + result.get(
                "error",
                "Video processing failed."
            )
        )

        return

    events = result["events"]
    weapon_found = len(events) > 0

    # --------------------------------------------------------
    # PHONE ALERTS
    # --------------------------------------------------------

    sms_results = []
    if weapon_found and SMS_ENABLED:
        for event in events:
            message = build_alert_message(
                "CCTV Video",
                float(event.get("confidence", 0.0)),
                float(event.get("time", 0.0))
            )
            sent, sms_status = send_phone_alert(message)
            sms_results.append(sms_status)
            if sent:
                break

    # --------------------------------------------------------
    # CREATE ALARM VIDEO
    # --------------------------------------------------------

    if weapon_found:

        st.warning(
            "🚨 Confirmed weapon detected. "
            "Adding alarm sound to output video..."
        )

        alarm_ok, alarm_error = create_alarm_video(
            silent_path,
            final_path,
            events
        )

        if not alarm_ok:

            st.error(
                "❌ Could not create alarm video."
            )

            with st.expander(
                "FFmpeg details"
            ):
                st.code(alarm_error)

            return

        output_path = final_path

        st.success(
            "🔊 Alarm sound added to the detected "
            "weapon event(s)."
        )

    else:

        # No weapon = no alarm.
        output_path = silent_path

    # --------------------------------------------------------
    # CONVERT FINAL OUTPUT FOR BROWSER
    # --------------------------------------------------------

    st.info(
        "🎬 Preparing browser-compatible output video..."
    )

    ok, conversion_error = encode_browser_video(
        output_path,
        browser_path
    )

    if not ok:

        st.error(
            "❌ Browser-compatible video conversion failed."
        )

        with st.expander(
            "FFmpeg details"
        ):
            st.code(conversion_error)

        return

    # --------------------------------------------------------
    # SIDE-BY-SIDE VIDEO
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("## 🎬 Input vs AI Detection")

    col1, col2 = st.columns(
        2,
        gap="large"
    )

    with col1:

        st.markdown("### 📥 Input CCTV Video")

        with open(
            input_path,
            "rb"
        ) as f:
            input_bytes = f.read()

        st.video(
            input_bytes
        )

    with col2:

        if weapon_found:
            st.markdown(
                "### 🚨 Output — Weapon Detected"
            )
        else:
            st.markdown(
                "### ✅ Output — No Weapon Detected"
            )

        with open(
            browser_path,
            "rb"
        ) as f:
            output_bytes = f.read()

        st.video(
            output_bytes
        )

        st.download_button(
            "⬇️ Download Output Video",
            data=output_bytes,
            file_name="WeaponGuard_AI_Result.mp4",
            mime="video/mp4",
            use_container_width=True
        )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    st.markdown("---")

    if weapon_found:

        st.markdown(
            f"""
            <div class="wg-danger">
                🚨 WEAPON DETECTED IN CCTV VIDEO<br><br>
                🔊 Alarm sound is included in the output video<br>
                🎯 Highest confidence:
                {result["highest_confidence"]:.2%}<br>
                ⏱️ Confirmed event(s):
                {len(events)}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### 🔊 Test Alarm Sound")

        show_alarm_player()

        if SMS_ENABLED:
            if sms_results and any("successfully" in x.lower() for x in sms_results):
                st.success("📱 Phone alert: SMS sent to the configured security/owner number.")
            elif sms_results:
                st.warning("📱 Phone alert: " + sms_results[0])

    else:

        st.markdown(
            """
            <div class="wg-safe">
                ✅ NO WEAPON DETECTED
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# LIVE CAMERA
# ============================================================

def _send_twilio_sms_direct(message):
    """Thread-safe Twilio SMS sender for the WebRTC callback."""
    if not SMS_ENABLED:
        return False

    if not (
        TWILIO_ACCOUNT_SID
        and TWILIO_AUTH_TOKEN
        and TWILIO_FROM_NUMBER
        and ALERT_TO_NUMBER
    ):
        return False

    url = (
        "https://api.twilio.com/2010-04-01/Accounts/"
        f"{TWILIO_ACCOUNT_SID}/Messages.json"
    )

    payload = urllib.parse.urlencode({
        "From": TWILIO_FROM_NUMBER,
        "To": ALERT_TO_NUMBER,
        "Body": message[:1500]
    }).encode("utf-8")

    credentials = f"{TWILIO_ACCOUNT_SID}:{TWILIO_AUTH_TOKEN}".encode("utf-8")
    auth = base64.b64encode(credentials).decode("ascii")

    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/x-www-form-urlencoded"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def live_detection(
    confidence_threshold,
    required_frames
):
    """Browser webcam detection using WebRTC.

    cv2.VideoCapture(0) cannot access the user's webcam when the app is
    deployed remotely. WebRTC requests the camera from the user's browser.
    """

    st.subheader("📹 Live Camera Detection")

    if not WEBRTC_AVAILABLE:
        st.error(
            "❌ Browser webcam support is not installed. "
            "Add streamlit-webrtc and av to requirements.txt, then redeploy."
        )
        return

    st.info(
        "Click START below. Your browser will ask for camera permission. "
        "Choose Allow to use your webcam."
    )

    state = {
        "consecutive": 0,
        "was_confirmed": False,
        "last_log": 0.0,
        "last_sms": 0.0,
        "frame_count": 0,
    }

    inference_lock = threading.Lock()

    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        state["frame_count"] += 1

        # Run YOLO on alternate frames to keep the browser feed responsive.
        if state["frame_count"] % 2 == 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        if not inference_lock.acquire(blocking=False):
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        try:
            output, detections, highest = detect_weapons(
                img,
                confidence_threshold
            )
        finally:
            inference_lock.release()

        if detections:
            state["consecutive"] += 1
        else:
            state["consecutive"] = 0

        confirmed = state["consecutive"] >= required_frames

        if confirmed:
            output = add_banner(
                output,
                "🚨 WEAPON CONFIRMED",
                (0, 0, 180)
            )

            now = time.time()

            if now - state["last_log"] >= 5:
                save_detection_log(
                    "Live Camera",
                    highest,
                    len(detections)
                )
                state["last_log"] = now

            if (
                not state["was_confirmed"]
                and SMS_ENABLED
                and now - state["last_sms"] >= SMS_COOLDOWN_SECONDS
            ):
                message = build_alert_message(
                    "Live Camera",
                    highest
                )
                state["last_sms"] = now
                threading.Thread(
                    target=_send_twilio_sms_direct,
                    args=(message,),
                    daemon=True
                ).start()

        elif detections:
            output = add_banner(
                output,
                "VERIFYING...",
                (0, 120, 220)
            )
        else:
            output = add_banner(
                output,
                "NO WEAPON DETECTED",
                (0, 80, 0)
            )

        state["was_confirmed"] = confirmed
        return av.VideoFrame.from_ndarray(output, format="bgr24")

    try:
        ctx = webrtc_streamer(
            key="weaponguard-browser-live-v2",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 15, "max": 20},
                },
                "audio": False,
            },
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]}
                ]
            },
            async_processing=False,
            media_stream_constraints_timeout=10,
        )
    except TypeError:
        ctx = webrtc_streamer(
            key="weaponguard-browser-live-v2",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 15, "max": 20},
                },
                "audio": False,
            },
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]}
                ]
            },
            async_processing=False,
        )

    if ctx.state.playing:
        st.success(
            "🟢 Webcam is connected. The browser camera feed is being analyzed."
        )
    else:
        st.caption(
            "If the camera does not start, click START and choose Allow "
            "when Chrome asks for camera permission."
        )


# ============================================================
# LOGS
# ============================================================

def show_logs():

    st.subheader("📋 Detection Logs")

    if not os.path.exists(LOG_FILE):

        st.info("No detection logs yet.")
        return

    try:

        with open(
            LOG_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            logs = json.load(f)

        if isinstance(logs, dict):
            logs = logs.get(
                "detections",
                [logs]
            )

        if not logs:

            st.info("No detection logs yet.")
            return

        st.dataframe(
            list(reversed(logs)),
            use_container_width=True
        )

    except Exception as e:

        st.warning(
            f"Could not read logs: {e}"
        )


# ============================================================
# DETECTION PAGE
# ============================================================

def detection_page():

    with st.sidebar:

        st.markdown(
            """
            <div class="wg-sidebar-title">
                🚨 WeaponGuard AI
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "⬅️ Back to Home",
            use_container_width=True
        ):

            st.session_state.page = "home"
            st.rerun()

        st.divider()

        st.markdown("### ⚙️ Detection Settings")

        confidence = st.slider(
            "🎯 Confidence Threshold",
            min_value=0.20,
            max_value=0.95,
            value=DEFAULT_CONFIDENCE,
            step=0.05
        )

        required_frames = st.slider(
            "🎞️ Required Consecutive Frames",
            min_value=1,
            max_value=15,
            value=DEFAULT_REQUIRED_FRAMES,
            step=1
        )

        st.caption(
            "Higher confidence + more consecutive frames "
            "can reduce false alarms."
        )

        st.divider()

        st.markdown("### 📱 Phone Alert")
        if SMS_ENABLED:
            st.success("SMS alerts are enabled")
            st.caption("Alerts use the configured Twilio account and phone number.")
        else:
            st.info("SMS alerts are disabled. Configure the environment variables to enable them.")

        st.divider()

        st.markdown("### 🎨 Appearance")

        theme = st.selectbox(
            "Theme",
            ["System", "Light", "Dark"],
            index=["System", "Light", "Dark"].index(
                st.session_state.theme
            ),
            label_visibility="collapsed"
        )

        if theme != st.session_state.theme:

            st.session_state.theme = theme
            st.rerun()

        st.divider()

        st.markdown("### 🤖 Model")
        st.write("YOLO")
        st.write("🎯 Classes: Person / Weapon")
        st.write("📷 Image / Video / Live Camera")

    st.markdown(
        '<div class="wg-title">🚨 WeaponGuard AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="wg-subtitle">'
        'AI-powered weapon detection'
        '</div>',
        unsafe_allow_html=True
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📷 Image Detection",
            "🎥 Video Detection",
            "📹 Live Detection",
            "📋 Logs"
        ]
    )

    with tab1:
        image_detection(confidence)

    with tab2:
        video_detection(
            confidence,
            required_frames
        )

    with tab3:
        live_detection(
            confidence,
            required_frames
        )

    with tab4:
        show_logs()


# ============================================================
# RUN
# ============================================================

if st.session_state.page == "home":
    home_page()
else:
    detection_page()
