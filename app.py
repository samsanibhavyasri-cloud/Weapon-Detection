import streamlit as st
import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
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
# CONSTANTS
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


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* -------------------------------------------------------
       GENERAL
    ------------------------------------------------------- */

    .stApp {
        background-color: #0e1117;
    }

    [data-testid="stSidebar"] {
        background-color: #151923;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 25px;
    }

    /* -------------------------------------------------------
       HEADINGS
    ------------------------------------------------------- */

    h1 {
        font-weight: 800 !important;
    }

    h2 {
        font-weight: 700 !important;
    }

    h3 {
        font-weight: 700 !important;
    }

    /* -------------------------------------------------------
       BUTTONS
    ------------------------------------------------------- */

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 45px;
    }

    /* Main red buttons */
    div.stButton > button[kind="primary"] {
        background-color: #ff4b4b;
        border-color: #ff4b4b;
        color: white;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #ff3333;
        border-color: #ff3333;
        color: white;
    }

    /* -------------------------------------------------------
       FILE UPLOADER
    ------------------------------------------------------- */

    [data-testid="stFileUploader"] {
        background-color: #262832;
        border-radius: 10px;
        padding: 10px;
    }

    /* -------------------------------------------------------
       TABS
    ------------------------------------------------------- */

    button[data-baseweb="tab"] {
        font-size: 16px;
        font-weight: 600;
    }

    /* -------------------------------------------------------
       INFO BOX
    ------------------------------------------------------- */

    .info-card {
        background-color: #181c25;
        border-radius: 12px;
        padding: 25px;
        border: 1px solid #292d38;
    }

    /* -------------------------------------------------------
       HOME DESCRIPTION
    ------------------------------------------------------- */

    .description-text {
        text-align: center;
        font-size: 18px;
        line-height: 2;
        padding: 20px 70px;
    }

    /* -------------------------------------------------------
       DETECTION STATUS
    ------------------------------------------------------- */

    .weapon-alert {
        background-color: #8b0000;
        color: white;
        padding: 18px;
        border-radius: 10px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
    }

    .safe-alert {
        background-color: #123d20;
        color: #7dff9b;
        padding: 18px;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "home"


# ============================================================
# DETECTION LOG
# ============================================================

def save_detection_log(source, confidence):

    data = {
        "timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "source": source,
        "confidence": round(
            float(confidence),
            3
        )
    }

    try:

        if os.path.exists(LOG_FILE):

            with open(
                LOG_FILE,
                "r",
                encoding="utf-8"
            ) as f:

                logs = json.load(f)

            # Fix dictionary problem
            if isinstance(logs, dict):
                logs = [logs]

            elif not isinstance(logs, list):
                logs = []

        else:

            logs = []

    except Exception:

        logs = []

    logs.append(data)

    # Keep latest 100 logs
    logs = logs[-100:]

    try:

        with open(
            LOG_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                logs,
                f,
                indent=4
            )

    except Exception:
        pass


# ============================================================
# LOAD LOGS
# ============================================================

def load_detection_logs():

    if not os.path.exists(LOG_FILE):
        return []

    try:

        with open(
            LOG_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            logs = json.load(f)

        if isinstance(logs, dict):
            logs = [logs]

        if not isinstance(logs, list):
            return []

        return logs

    except Exception:

        return []


# ============================================================
# DETECT WEAPON
# ============================================================

def detect_frame(
    frame,
    confidence=0.30
):

    output = frame.copy()

    weapon_found = False
    highest_confidence = 0.0
    weapon_count = 0

    results = model(
        frame,
        conf=confidence,
        verbose=False
    )

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            conf = float(
                box.conf[0]
            )

            class_id = int(
                box.cls[0]
            )

            class_name = str(
                model.names[class_id]
            )

            if class_name.lower() == "weapon":

                weapon_found = True

                weapon_count += 1

                highest_confidence = max(
                    highest_confidence,
                    conf
                )

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                # Red bounding box
                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    3
                )

                label = (
                    f"WEAPON {conf:.2f}"
                )

                cv2.putText(
                    output,
                    label,
                    (
                        x1,
                        max(y1 - 10, 30)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

    return (
        output,
        weapon_found,
        highest_confidence,
        weapon_count
    )


# ============================================================
# HOME PAGE
# ============================================================

def home_page():

    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------

    with st.sidebar:

        st.markdown(
            """
            <h2 style="
                font-size: 26px;
                line-height: 1.35;
            ">
            Artificial Intelligence<br>
            Career for Women (AICW)
            </h2>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")

        st.markdown(
            "### 🎓 College"
        )

        st.write(
            COLLEGE_NAME
        )

        st.markdown("---")

        st.markdown(
            "### 👥 Team Members"
        )

        st.markdown(
            f"⭐ **{TEAM_LEAD} — Team Lead**"
        )

        for member in TEAM_MEMBERS:

            st.markdown(
                f"• {member} — Team Member"
            )

        st.markdown("---")

        st.markdown(
            "### 🧑‍🏫 Project Guide"
        )

        st.write(
            GUIDE_NAME
        )

    # --------------------------------------------------------
    # MAIN CONTENT
    # --------------------------------------------------------

    st.markdown(
        """
        <div style="
            text-align:center;
            margin-top:50px;
        ">
            <h1 style="font-size:48px;">
                🚨 WeaponGuard AI
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h2 style="
            text-align:center;
            margin-top:25px;
        ">
            Description
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="description-text">

        WeaponGuard AI is an intelligent weapon detection
        system designed to improve security through automated
        image and CCTV video analysis. The system uses
        Artificial Intelligence and YOLO-based object detection
        to identify weapons in uploaded media and live camera
        feeds. When a weapon is detected, the system highlights
        the detected object and provides an alert. By reducing
        the need for continuous manual monitoring, the solution
        helps security personnel identify potential threats
        quickly and respond more effectively.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

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

def image_detection(
    confidence_threshold
):

    st.markdown(
        "## 📷 Upload Image"
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp",
            "webp"
        ],
        key="image_upload"
    )

    if uploaded_file is None:
        return

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    image_array = np.array(
        image
    )

    # For image detection use a low threshold
    # so the trained model has a chance to detect.
    detection_confidence = min(
        confidence_threshold,
        0.30
    )

    (
        output,
        weapon_found,
        highest_confidence,
        weapon_count
    ) = detect_frame(
        image_array,
        detection_confidence
    )

    output_rgb = cv2.cvtColor(
        output,
        cv2.COLOR_BGR2RGB
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### Original Image"
        )

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.markdown(
            "### Detection Result"
        )

        st.image(
            output_rgb,
            use_container_width=True
        )

    st.markdown("---")

    if weapon_found:

        st.markdown(
            f"""
            <div class="weapon-alert">
            🚨 WEAPON DETECTED<br>
            Weapons: {weapon_count}<br>
            Confidence: {highest_confidence:.2%}
            </div>
            """,
            unsafe_allow_html=True
        )

        save_detection_log(
            "Image Detection",
            highest_confidence
        )

        if os.path.exists(
            ALERT_SOUND
        ):

            st.audio(
                ALERT_SOUND,
                format="audio/mp3"
            )

    else:

        st.markdown(
            """
            <div class="safe-alert">
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

    st.markdown(
        "## 🎥 Upload CCTV Video"
    )

    uploaded_video = st.file_uploader(
        "Choose a CCTV video",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv"
        ],
        key="video_upload"
    )

    if uploaded_video is None:
        return

    temp_video = "temp_input_video.mp4"

    with open(
        temp_video,
        "wb"
    ) as f:

        f.write(
            uploaded_video.read()
        )

    st.success(
        "✅ Video uploaded successfully."
    )

    if st.button(
        "▶️ START VIDEO DETECTION",
        type="primary",
        use_container_width=True,
        key="start_video"
    ):

        cap = cv2.VideoCapture(
            temp_video
        )

        if not cap.isOpened():

            st.error(
                "❌ Could not open video."
            )

            return

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        frame_placeholder = st.empty()

        progress = st.progress(0)

        status = st.empty()

        consecutive_count = 0

        highest_confidence = 0.0

        logged = False

        frame_number = 0

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            (
                output,
                weapon_found,
                confidence,
                weapon_count
            ) = detect_frame(
                frame,
                confidence_threshold
            )

            if weapon_found:

                consecutive_count += 1

                highest_confidence = max(
                    highest_confidence,
                    confidence
                )

            else:

                consecutive_count = 0

            if (
                consecutive_count
                >= required_frames
            ):

                cv2.rectangle(
                    output,
                    (0, 0),
                    (
                        output.shape[1],
                        65
                    ),
                    (0, 0, 255),
                    -1
                )

                cv2.putText(
                    output,
                    "WEAPON DETECTED",
                    (20, 43),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    3
                )

                status.error(
                    f"🚨 WEAPON DETECTED | "
                    f"Confidence: "
                    f"{confidence:.2%}"
                )

                if not logged:

                    save_detection_log(
                        "Video Detection",
                        confidence
                    )

                    logged = True

            else:

                cv2.rectangle(
                    output,
                    (0, 0),
                    (
                        output.shape[1],
                        55
                    ),
                    (0, 0, 0),
                    -1
                )

                cv2.putText(
                    output,
                    "NO WEAPON DETECTED",
                    (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (0, 255, 0),
                    2
                )

                status.success(
                    "✅ NO WEAPON DETECTED"
                )

            output_rgb = cv2.cvtColor(
                output,
                cv2.COLOR_BGR2RGB
            )

            frame_placeholder.image(
                output_rgb,
                channels="RGB",
                use_container_width=True
            )

            if total_frames > 0:

                progress.progress(
                    min(
                        frame_number /
                        total_frames,
                        1.0
                    )
                )

        cap.release()

        progress.progress(1.0)

        st.success(
            "✅ Video detection completed."
        )


# ============================================================
# LIVE DETECTION
# ============================================================

def live_detection():

    st.markdown(
        "## 📹 Live Camera Detection"
    )

    st.info(
        """
        **Live Detection using OpenCV**

        Click **START LIVE DETECTION** below.
        Your webcam will open in a separate window.

        Press **Q** or **ESC** inside the camera
        window to stop.
        """
    )

    live_confidence = st.slider(
        "Live Detection Confidence",
        min_value=0.05,
        max_value=0.95,
        value=0.20,
        step=0.05,
        key="live_confidence"
    )

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    if st.button(
        "📹 START LIVE DETECTION",
        type="primary",
        use_container_width=True,
        key="start_live_detection"
    ):

        camera = cv2.VideoCapture(
            0,
            cv2.CAP_DSHOW
        )

        # Try second camera if first fails
        if not camera.isOpened():

            camera.release()

            camera = cv2.VideoCapture(
                1,
                cv2.CAP_DSHOW
            )

        if not camera.isOpened():

            st.error(
                """
                ❌ Could not open webcam.

                Please check:

                • Camera permission
                • Camera connection
                • Windows Camera app
                • Another application using the camera
                """
            )

            return

        camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            640
        )

        camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            480
        )

        camera.set(
            cv2.CAP_PROP_FPS,
            30
        )

        st.success(
            "🟢 Live camera started."
        )

        st.info(
            "Press Q or ESC in the camera window to stop."
        )

        last_log_time = 0

        while True:

            success, frame = camera.read()

            if not success:

                st.error(
                    "❌ Could not read camera frame."
                )

                break

            (
                output,
                weapon_found,
                highest_confidence,
                weapon_count
            ) = detect_frame(
                frame,
                live_confidence
            )

            # ------------------------------------------------
            # WEAPON DETECTED
            # ------------------------------------------------

            if weapon_found:

                cv2.rectangle(
                    output,
                    (0, 0),
                    (
                        output.shape[1],
                        65
                    ),
                    (0, 0, 255),
                    -1
                )

                cv2.putText(
                    output,
                    "WEAPON DETECTED",
                    (20, 43),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    3
                )

                current_time = time.time()

                if (
                    current_time -
                    last_log_time
                    >= 5
                ):

                    save_detection_log(
                        "Live Camera",
                        highest_confidence
                    )

                    last_log_time = current_time

            # ------------------------------------------------
            # NO WEAPON
            # ------------------------------------------------

            else:

                cv2.rectangle(
                    output,
                    (0, 0),
                    (
                        output.shape[1],
                        55
                    ),
                    (0, 0, 0),
                    -1
                )

                cv2.putText(
                    output,
                    "NO WEAPON DETECTED",
                    (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (0, 255, 0),
                    2
                )

            # ------------------------------------------------
            # CAMERA WINDOW
            # ------------------------------------------------

            cv2.imshow(
                "WeaponGuard AI - LIVE DETECTION",
                output
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == 27:
                break

        camera.release()

        cv2.destroyAllWindows()

        for _ in range(3):
            cv2.waitKey(1)

        st.success(
            "🟢 Live Detection stopped."
        )


# ============================================================
# DETECTION LOGS
# ============================================================

def detection_logs():

    st.markdown(
        "## 📋 Detection Logs"
    )

    logs = load_detection_logs()

    if not logs:

        st.info(
            "No detection logs available."
        )

        return

    logs = logs[::-1]

    st.dataframe(
        logs,
        use_container_width=True
    )


# ============================================================
# DETECTION PAGE
# ============================================================

def detection_page():

    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------

    with st.sidebar:

        st.markdown(
            """
            <h1 style="
                font-size: 27px;
                margin-top: 20px;
            ">
            WeaponGuard AI
            </h1>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "⬅️ Back",
            use_container_width=True
        ):

            st.session_state.page = "home"

            st.rerun()

        st.markdown("---")

        st.markdown(
            "### ⚙️ Detection Settings"
        )

        confidence_threshold = st.slider(
            "🎯 Confidence Threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.60,
            step=0.05
        )

        required_frames = st.slider(
            "🎞️ Required Consecutive Frames",
            min_value=1,
            max_value=20,
            value=5,
            step=1
        )

        st.markdown("---")

        st.markdown(
            "### Model"
        )

        st.markdown(
            "⚙️ **Model:** YOLO"
        )

        st.markdown(
            "🎯 **Detection:** Weapon"
        )

        st.markdown(
            "📷 **Input:** Image / CCTV Video / Live Camera"
        )

    # --------------------------------------------------------
    # MAIN HEADER
    # --------------------------------------------------------

    st.markdown(
        """
        <div style="
            text-align:center;
            margin-top:20px;
        ">
            <h1 style="font-size:48px;">
                🚨 WeaponGuard AI
            </h1>

            <p style="
                font-size:18px;
                color:#aab0bd;
            ">
                AI-powered weapon detection using YOLO
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # --------------------------------------------------------
    # TABS
    # --------------------------------------------------------

    tab1, tab2, tab3 = st.tabs(
        [
            "📷 Image Detection",
            "🎥 Video Detection",
            "📹 Live Detection"
        ]
    )

    # --------------------------------------------------------
    # IMAGE TAB
    # --------------------------------------------------------

    with tab1:

        image_detection(
            confidence_threshold
        )

    # --------------------------------------------------------
    # VIDEO TAB
    # --------------------------------------------------------

    with tab2:

        video_detection(
            confidence_threshold,
            required_frames
        )

    # --------------------------------------------------------
    # LIVE TAB
    # --------------------------------------------------------

    with tab3:

        live_detection()


# ============================================================
# MAIN
# ============================================================

if st.session_state.page == "home":

    home_page()

else:

    detection_page()
