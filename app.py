import streamlit as st
import cv2
import os
import math
import tempfile
import json
import subprocess
from datetime import datetime

import numpy as np
from PIL import Image
from ultralytics import YOLO


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="WeaponGuard AI",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "best.pt"

# Used mainly for VIDEO detection
CONFIDENCE_THRESHOLD = 0.30

# Number of consecutive frames required for video confirmation
REQUIRED_CONSECUTIVE_FRAMES = 3

TEAM_LEAD = "S. Nagasindhu"

TEAM_MEMBERS = [
    "S. Bhavyasri",
    "S. Manasa",
    "S. Anusha"
]

GUIDE_NAME = "Mr. Abdul Aziz MD"

COLLEGE_NAME = "VSM College of Engineering"


# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "home"


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0e1117;
    }

    section[data-testid="stSidebar"] {
        background-color: #151923;
    }

    .sidebar-aicw {
        text-align: center;
        font-size: 25px;
        font-weight: 800;
        color: white;
        line-height: 1.3;
        padding: 20px 5px 25px 5px;
    }

    .sidebar-heading {
        color: #4da6ff;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .detection-title {
        text-align: center;
        font-size: 45px;
        font-weight: 800;
        color: white;
        margin-top: 20px;
        margin-bottom: 5px;
    }

    .detection-subtitle {
        text-align: center;
        color: #aaaaaa;
        font-size: 17px;
        margin-bottom: 25px;
    }

    .detection-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 15px;
    }

    .safe {
        background-color: #123d24;
        color: #50fa7b;
    }

    .danger {
        background-color: #4a1111;
        color: #ff5555;
    }

    .video-heading {
        text-align: center;
        font-size: 20px;
        font-weight: 700;
        color: white;
        margin-bottom: 10px;
    }

    div.stButton > button {
        border-radius: 10px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def box_center(box):
    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )


def distance(box1, box2):
    c1 = box_center(box1)
    c2 = box_center(box2)

    return math.sqrt(
        (c1[0] - c2[0]) ** 2 +
        (c1[1] - c2[1]) ** 2
    )


# =========================================================
# SAVE DETECTION DATA
# =========================================================

def save_detection_data(
    detection_type,
    weapon_count,
    total_objects=0
):

    data_file = "detection_log.json"

    detection_record = {
        "timestamp": datetime.now().isoformat(),
        "type": detection_type,
        "weapons_detected": weapon_count,
        "total_objects": total_objects
    }

    try:

        if os.path.exists(data_file):

            with open(data_file, "r") as file:
                data = json.load(file)

        else:

            data = {
                "detections": []
            }

        data["detections"].append(
            detection_record
        )

        if len(data["detections"]) > 100:
            data["detections"] = data["detections"][-100:]

        with open(data_file, "w") as file:

            json.dump(
                data,
                file,
                indent=2
            )

    except Exception:
        pass


# =========================================================
# LOAD YOLO MODEL
# =========================================================

@st.cache_resource
def load_model():

    return YOLO(MODEL_PATH)


# =========================================================
# PLAY ALERT
# =========================================================

def play_alert():

    alert_path = "alert.mp3"

    if os.path.exists(alert_path):

        with open(
            alert_path,
            "rb"
        ) as audio_file:

            st.audio(
                audio_file.read(),
                format="audio/mp3"
            )


# =========================================================
# CONVERT VIDEO FOR BROWSER
# =========================================================

def convert_video_for_browser(input_path):

    converted_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix="_browser.mp4"
    )

    converted_file.close()

    output_path = converted_file.name

    try:

        command = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            output_path
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300
        )

        if (
            result.returncode == 0
            and os.path.exists(output_path)
        ):

            return output_path

    except Exception:
        pass

    return input_path


# =========================================================
# HOME PAGE
# =========================================================

def home_page():

    # =====================================================
    # SIDEBAR
    # =====================================================

    with st.sidebar:

        st.title(
            "Artificial Intelligence Career for Women (AICW)"
        )

        st.divider()

        st.subheader("🎓 College")

        st.write(
            COLLEGE_NAME
        )

        st.divider()

        st.subheader("👥 Team Members")

        st.markdown(
            "**⭐ S. Nagasindhu — Team Lead**"
        )

        st.write(
            "• S. Bhavyasri — Team Member"
        )

        st.write(
            "• S. Manasa — Team Member"
        )

        st.write(
            "• S. Anusha — Team Member"
        )

        st.divider()

        st.subheader("👨‍🏫 Project Guide")

        st.write(
            GUIDE_NAME
        )

    # =====================================================
    # MAIN CONTENT
    # =====================================================

    st.markdown(
        "<h1 style='text-align:center;'>🚨 WeaponGuard AI</h1>",
        unsafe_allow_html=True
    )

    st.write("")

    st.markdown(
        "<h2 style='text-align:center;'>Description</h2>",
        unsafe_allow_html=True
    )

    st.write("")

    st.markdown(
        """
        <div style="
            text-align:center;
            font-size:18px;
            line-height:1.8;
            max-width:900px;
            margin:auto;
        ">

        WeaponGuard AI is an intelligent weapon detection system
        designed to improve security through automated image and
        CCTV video analysis.

        The system uses Artificial Intelligence and YOLO-based
        object detection to identify weapons in uploaded media.

        When a weapon is detected, the system highlights the
        detected object and provides an alert sound.

        By reducing the need for continuous manual monitoring,
        the solution helps security personnel identify potential
        threats quickly and respond more effectively.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    st.write("")
    st.write("")

    if st.button(
        "➡️ NEXT",
        use_container_width=True,
        type="primary"
    ):

        st.session_state.page = "detection"

        st.rerun()


# =========================================================
# DETECTION PAGE
# =========================================================

def detection_page():

    # =====================================================
    # SIDEBAR
    # =====================================================

    with st.sidebar:

        st.markdown(
            """
            <div class="sidebar-aicw">
                🚨 WeaponGuard AI
            </div>
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
            """
            <div class="sidebar-heading">
                ⚙️ Detection Settings
            </div>
            """,
            unsafe_allow_html=True
        )

        confidence_threshold = st.slider(
            "🎯 Video Confidence Threshold",
            min_value=0.10,
            max_value=1.00,
            value=0.30,
            step=0.05
        )

        required_frames = st.slider(
            "🎞️ Required Consecutive Frames",
            min_value=1,
            max_value=15,
            value=3,
            step=1
        )

        st.markdown("---")

        st.write("🤖 **Model:** YOLO")

        st.write("🎯 **Detection:** Weapon")

        st.write(
            "📷 **Input:** Image / CCTV Video"
        )

    # =====================================================
    # CHECK MODEL
    # =====================================================

    if not os.path.exists(MODEL_PATH):

        st.error(
            "❌ best.pt not found!"
        )

        st.info(
            "Please keep best.pt in the same folder "
            "as app.py."
        )

        st.stop()

    # =====================================================
    # LOAD MODEL
    # =====================================================

    model = load_model()

    # =====================================================
    # HEADER
    # =====================================================

    st.markdown(
        """
        <div class="detection-title">
            🚨 WeaponGuard AI
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="detection-subtitle">
            AI-powered weapon detection using YOLO
        </div>
        """,
        unsafe_allow_html=True
    )

    # =====================================================
    # TABS
    # =====================================================

    tab1, tab2 = st.tabs(
        [
            "📷 Image Detection",
            "🎥 Video Detection"
        ]
    )


    # =====================================================
    # IMAGE DETECTION
    # =====================================================

    with tab1:

        st.subheader(
            "📷 Upload Image"
        )

        uploaded_image = st.file_uploader(
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

        if uploaded_image is not None:

            try:

                image = Image.open(
                    uploaded_image
                ).convert("RGB")

            except Exception as e:

                st.error(
                    f"❌ Could not read image: {e}"
                )

                st.stop()

            image_array = np.array(
                image
            )

            st.image(
                image_array,
                caption="📥 Input Image",
                use_container_width=True
            )

            st.write("")

            if st.button(
                "🔍 Detect Weapon",
                use_container_width=True,
                type="primary",
                key="image_detect"
            ):

                with st.spinner(
                    "🔎 Analyzing image..."
                ):

                    # -------------------------------------
                    # CONVERT IMAGE
                    # -------------------------------------

                    image_bgr = cv2.cvtColor(
                        image_array,
                        cv2.COLOR_RGB2BGR
                    )

                    # -------------------------------------
                    # IMPORTANT
                    #
                    # We use LOW confidence here.
                    # The sidebar video slider does NOT
                    # control image detection.
                    # -------------------------------------

                    results = model(
                        image_bgr,
                        conf=0.01,
                        verbose=False
                    )

                    output_image = image_bgr.copy()

                    weapon_count = 0

                    total_objects = 0

                    weapon_confidences = []

                    # -------------------------------------
                    # PROCESS YOLO RESULTS
                    # -------------------------------------

                    for result in results:

                        if result.boxes is None:
                            continue

                        for box in result.boxes:

                            confidence = float(
                                box.conf[0]
                            )

                            class_id = int(
                                box.cls[0]
                            )

                            class_name = str(
                                model.names[class_id]
                            )

                            total_objects += 1

                            # ---------------------------------
                            # WEAPON DETECTION
                            # ---------------------------------

                            if (
                                class_name.lower()
                                == "weapon"
                            ):

                                weapon_count += 1

                                weapon_confidences.append(
                                    confidence
                                )

                                # -----------------------------
                                # BOX
                                # -----------------------------

                                x1, y1, x2, y2 = map(
                                    int,
                                    box.xyxy[0]
                                )

                                cv2.rectangle(
                                    output_image,
                                    (x1, y1),
                                    (x2, y2),
                                    (0, 0, 255),
                                    4
                                )

                                # -----------------------------
                                # LABEL
                                # -----------------------------

                                label = (
                                    f"WEAPON "
                                    f"{confidence:.2f}"
                                )

                                cv2.putText(
                                    output_image,
                                    label,
                                    (
                                        x1,
                                        max(
                                            y1 - 10,
                                            30
                                        )
                                    ),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (0, 0, 255),
                                    2
                                )

                    # -------------------------------------
                    # CONVERT OUTPUT
                    # -------------------------------------

                    output_rgb = cv2.cvtColor(
                        output_image,
                        cv2.COLOR_BGR2RGB
                    )

                    # -------------------------------------
                    # SHOW RESULT
                    # -------------------------------------

                    st.markdown("---")

                    st.subheader(
                        "📤 Detection Result"
                    )

                    st.image(
                        output_rgb,
                        caption="Weapon Detection Output",
                        use_container_width=True
                    )

                    # -------------------------------------
                    # SAVE LOG
                    # -------------------------------------

                    save_detection_data(
                        "image",
                        weapon_count,
                        total_objects
                    )

                    # -------------------------------------
                    # DETECTION RESULT
                    # -------------------------------------

                    if weapon_count > 0:

                        highest_confidence = max(
                            weapon_confidences
                        )

                        st.markdown(
                            f"""
                            <div class="detection-box danger">
                                🚨 {weapon_count}
                                WEAPON(S) DETECTED
                                <br>
                                <span style="font-size:18px;">
                                Highest Confidence:
                                {highest_confidence:.2f}
                                </span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        st.markdown(
                            "### 🔊 Weapon Detection Alert"
                        )

                        play_alert()

                    else:

                        st.markdown(
                            """
                            <div class="detection-box safe">
                                ✅ NO WEAPON DETECTED
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        st.info(
                            "The model did not find a weapon "
                            "in this image."
                        )

                    # -------------------------------------
                    # DOWNLOAD IMAGE
                    # -------------------------------------

                    output_pil = Image.fromarray(
                        output_rgb
                    )

                    image_temp = tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".png"
                    )

                    output_pil.save(
                        image_temp.name
                    )

                    image_temp.close()

                    with open(
                        image_temp.name,
                        "rb"
                    ) as file:

                        st.download_button(
                            "⬇️ Download Output Image",
                            data=file.read(),
                            file_name="weapon_detection_output.png",
                            mime="image/png",
                            use_container_width=True
                        )


    # =====================================================
    # VIDEO DETECTION
    # =====================================================

    with tab2:

        st.subheader(
            "🎥 Upload CCTV Video"
        )

        uploaded_video = st.file_uploader(
            "Choose CCTV video",
            type=[
                "mp4",
                "avi",
                "mov",
                "mkv"
            ],
            key="video_upload"
        )

        if uploaded_video is not None:

            # -----------------------------------------
            # SAVE INPUT VIDEO
            # -----------------------------------------

            input_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            input_file.write(
                uploaded_video.read()
            )

            input_file.close()

            input_video_path = input_file.name

            # -----------------------------------------
            # INPUT VIDEO
            # -----------------------------------------

            st.markdown(
                """
                <div class="video-heading">
                    📥 Input CCTV Video
                </div>
                """,
                unsafe_allow_html=True
            )

            with open(
                input_video_path,
                "rb"
            ) as video_file:

                input_video_bytes = video_file.read()

            st.video(
                input_video_bytes
            )

            st.write("")

            # -----------------------------------------
            # START DETECTION
            # -----------------------------------------

            if st.button(
                "🔍 Start Weapon Detection",
                use_container_width=True,
                type="primary",
                key="video_detect"
            ):

                # -----------------------------------------
                # OUTPUT FILE
                # -----------------------------------------

                output_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                )

                output_video_path = (
                    output_file.name
                )

                output_file.close()

                # -----------------------------------------
                # OPEN VIDEO
                # -----------------------------------------

                cap = cv2.VideoCapture(
                    input_video_path
                )

                if not cap.isOpened():

                    st.error(
                        "❌ Could not open uploaded video."
                    )

                    st.stop()

                fps = cap.get(
                    cv2.CAP_PROP_FPS
                )

                if fps <= 0:
                    fps = 25

                width = int(
                    cap.get(
                        cv2.CAP_PROP_FRAME_WIDTH
                    )
                )

                height = int(
                    cap.get(
                        cv2.CAP_PROP_FRAME_HEIGHT
                    )
                )

                total_frames = int(
                    cap.get(
                        cv2.CAP_PROP_FRAME_COUNT
                    )
                )

                # -----------------------------------------
                # VIDEO WRITER
                # -----------------------------------------

                fourcc = cv2.VideoWriter_fourcc(
                    *"mp4v"
                )

                out = cv2.VideoWriter(
                    output_video_path,
                    fourcc,
                    fps,
                    (width, height)
                )

                # -----------------------------------------
                # VARIABLES
                # -----------------------------------------

                frame_number = 0

                consecutive_weapon_frames = {}

                last_boxes = {}

                confirmed_detections = 0

                detection_events = []

                # -----------------------------------------
                # PROGRESS
                # -----------------------------------------

                progress_bar = st.progress(0)

                status_text = st.empty()

                preview_placeholder = st.empty()

                # -----------------------------------------
                # PROCESS VIDEO
                # -----------------------------------------

                while True:

                    ret, frame = cap.read()

                    if not ret:
                        break

                    frame_number += 1

                    current_weapons = []

                    # -------------------------------------
                    # YOLO
                    # -------------------------------------

                    results = model(
                        frame,
                        conf=confidence_threshold,
                        verbose=False
                    )

                    # -------------------------------------
                    # FIND WEAPONS
                    # -------------------------------------

                    for result in results:

                        if result.boxes is None:
                            continue

                        for box in result.boxes:

                            confidence = float(
                                box.conf[0]
                            )

                            class_id = int(
                                box.cls[0]
                            )

                            class_name = str(
                                model.names[class_id]
                            )

                            if (
                                class_name.lower()
                                == "weapon"
                            ):

                                x1, y1, x2, y2 = map(
                                    int,
                                    box.xyxy[0]
                                )

                                current_weapons.append(
                                    {
                                        "box": (
                                            x1,
                                            y1,
                                            x2,
                                            y2
                                        ),
                                        "confidence": confidence
                                    }
                                )

                    # -------------------------------------
                    # TRACKING
                    # -------------------------------------

                    confirmed_weapon_boxes = []

                    new_last_boxes = {}

                    for weapon in current_weapons:

                        weapon_box = weapon["box"]

                        weapon_conf = weapon["confidence"]

                        best_match_id = None

                        best_match_distance = 150

                        for (
                            prev_id,
                            prev_box
                        ) in last_boxes.items():

                            movement = distance(
                                prev_box,
                                weapon_box
                            )

                            if movement < best_match_distance:

                                best_match_distance = movement

                                best_match_id = prev_id

                        # ---------------------------------
                        # EXISTING OBJECT
                        # ---------------------------------

                        if best_match_id is not None:

                            consecutive_weapon_frames[
                                best_match_id
                            ] = (
                                consecutive_weapon_frames.get(
                                    best_match_id,
                                    0
                                ) + 1
                            )

                            new_last_boxes[
                                best_match_id
                            ] = weapon_box

                            if (
                                consecutive_weapon_frames[
                                    best_match_id
                                ] >= required_frames
                            ):

                                confirmed_weapon_boxes.append(
                                    {
                                        "box": weapon_box,
                                        "confidence": weapon_conf,
                                        "id": best_match_id
                                    }
                                )

                        # ---------------------------------
                        # NEW OBJECT
                        # ---------------------------------

                        else:

                            new_id = max(
                                consecutive_weapon_frames.keys(),
                                default=-1
                            ) + 1

                            consecutive_weapon_frames[
                                new_id
                            ] = 1

                            new_last_boxes[
                                new_id
                            ] = weapon_box

                    last_boxes = new_last_boxes

                    # -------------------------------------
                    # DRAW DETECTIONS
                    # -------------------------------------

                    if confirmed_weapon_boxes:

                        confirmed_detections += len(
                            confirmed_weapon_boxes
                        )

                        for detection in confirmed_weapon_boxes:

                            x1, y1, x2, y2 = (
                                detection["box"]
                            )

                            confidence = (
                                detection["confidence"]
                            )

                            cv2.rectangle(
                                frame,
                                (x1, y1),
                                (x2, y2),
                                (0, 0, 255),
                                3
                            )

                            cv2.putText(
                                frame,
                                f"WEAPON {confidence:.2f}",
                                (
                                    x1,
                                    max(
                                        y1 - 10,
                                        30
                                    )
                                ),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 0, 255),
                                2
                            )

                            time_sec = (
                                frame_number / fps
                            )

                            detection_events.append(
                                {
                                    "time": round(
                                        time_sec,
                                        2
                                    ),
                                    "confidence": round(
                                        confidence,
                                        2
                                    )
                                }
                            )

                        cv2.putText(
                            frame,
                            (
                                f"!!! "
                                f"{len(confirmed_weapon_boxes)} "
                                f"WEAPON(S) DETECTED !!!"
                            ),
                            (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            3
                        )

                    else:

                        cv2.putText(
                            frame,
                            "No Weapon Detected",
                            (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 0),
                            2
                        )

                    # -------------------------------------
                    # WRITE FRAME
                    # -------------------------------------

                    out.write(
                        frame
                    )

                    # -------------------------------------
                    # LIVE PREVIEW
                    # -------------------------------------

                    preview_rgb = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )

                    preview_placeholder.image(
                        preview_rgb,
                        caption="🔄 Processing...",
                        use_container_width=True
                    )

                    # -------------------------------------
                    # PROGRESS
                    # -------------------------------------

                    if total_frames > 0:

                        progress = (
                            frame_number /
                            total_frames
                        )

                        progress_bar.progress(
                            min(progress, 1.0)
                        )

                    status_text.write(
                        f"Processing frame "
                        f"{frame_number} / "
                        f"{total_frames}"
                    )

                # -----------------------------------------
                # RELEASE VIDEO
                # -----------------------------------------

                cap.release()

                out.release()

                # -----------------------------------------
                # FINISH PROGRESS
                # -----------------------------------------

                progress_bar.progress(
                    1.0
                )

                status_text.success(
                    "✅ Video processing completed!"
                )

                preview_placeholder.empty()

                # -----------------------------------------
                # SAVE LOG
                # -----------------------------------------

                save_detection_data(
                    "video",
                    confirmed_detections,
                    total_frames
                )

                # -----------------------------------------
                # CONVERT VIDEO
                # -----------------------------------------

                browser_video_path = (
                    convert_video_for_browser(
                        output_video_path
                    )
                )

                # -----------------------------------------
                # RESULTS
                # -----------------------------------------

                st.markdown("---")

                st.subheader(
                    "📊 Detection Results"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        label="🎞️ Frames",
                        value=total_frames
                    )

                with col2:

                    st.metric(
                        label="🚨 Detections",
                        value=confirmed_detections
                    )

                with col3:

                    if confirmed_detections > 0:

                        st.error(
                            "🚨 WEAPON DETECTED"
                        )

                    else:

                        st.success(
                            "✅ NO WEAPON DETECTED"
                        )

                # -----------------------------------------
                # ALERT
                # -----------------------------------------

                if confirmed_detections > 0:

                    st.markdown(
                        "### 🔊 Weapon Detection Alert"
                    )

                    play_alert()

                # -----------------------------------------
                # DETECTION EVENTS
                # -----------------------------------------

                if detection_events:

                    st.markdown("---")

                    st.subheader(
                        "🚨 Detection Events"
                    )

                    displayed_events = (
                        detection_events[:50]
                    )

                    for event in displayed_events:

                        st.warning(
                            f"⏱️ Time: "
                            f"{event['time']} sec  |  "
                            f"Confidence: "
                            f"{event['confidence']}"
                        )

                # -----------------------------------------
                # INPUT / OUTPUT SIDE BY SIDE
                # -----------------------------------------

                st.markdown("---")

                st.subheader(
                    "🎥 Video Comparison"
                )

                video_col1, video_col2 = (
                    st.columns(2)
                )

                # -----------------------------------------
                # INPUT
                # -----------------------------------------

                with video_col1:

                    st.markdown(
                        """
                        <div class="video-heading">
                            📥 Input Video
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    with open(
                        input_video_path,
                        "rb"
                    ) as input_file:

                        input_video_bytes = (
                            input_file.read()
                        )

                    st.video(
                        input_video_bytes
                    )

                # -----------------------------------------
                # OUTPUT
                # -----------------------------------------

                with video_col2:

                    st.markdown(
                        """
                        <div class="video-heading">
                            📤 Output Video
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    if os.path.exists(
                        browser_video_path
                    ):

                        with open(
                            browser_video_path,
                            "rb"
                        ) as output_file:

                            output_video_bytes = (
                                output_file.read()
                            )

                        st.video(
                            output_video_bytes
                        )

                        st.download_button(
                            "⬇️ Download Processed Video",
                            data=output_video_bytes,
                            file_name=(
                                "weapon_detection_result.mp4"
                            ),
                            mime="video/mp4",
                            use_container_width=True
                        )

                # -----------------------------------------
                # CLEAN INPUT TEMP FILE
                # -----------------------------------------

                try:

                    if os.path.exists(
                        input_video_path
                    ):

                        os.remove(
                            input_video_path
                        )

                except Exception:
                    pass


# =========================================================
# PAGE ROUTING
# =========================================================

if st.session_state.page == "home":

    home_page()

else:

    detection_page()
