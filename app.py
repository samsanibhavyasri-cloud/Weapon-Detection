import streamlit as st
import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="WeaponGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_PATH = "best.pt"
ALERT_SOUND = "alert.mp3"
LOG_FILE = "detection_log.json"

TEAM_LEAD = "S. Nagasindhu"

TEAM_MEMBERS = [
    "S. Bhavyasri",
    "S. Manasa",
    "S. Anusha"
]

GUIDE_NAME = "Mr. Abdul Aziz MD"
COLLEGE_NAME = "VSM College of Engineering"


# ============================================================
# LOAD YOLO MODEL
# ============================================================

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


try:
    model = load_model()

except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "home"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🛡️ WeaponGuard AI")

    st.markdown(
        "### Artificial Intelligence Career for Women (AICW)"
    )

    st.markdown("---")

    st.markdown(
        f"**College:**  \n{COLLEGE_NAME}"
    )

    st.markdown(
        f"**Team Lead:**  \n{TEAM_LEAD}"
    )

    st.markdown("**Team Members:**")

    for member in TEAM_MEMBERS:
        st.markdown(f"- {member}")

    st.markdown(
        f"**Guide:**  \n{GUIDE_NAME}"
    )

    st.markdown("---")

    st.markdown("### ⚙️ Detection Settings")

    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.30,
        step=0.05
    )

    required_frames = st.slider(
        "Required Consecutive Frames",
        min_value=1,
        max_value=10,
        value=3,
        step=1
    )


# ============================================================
# SAVE DETECTION LOG
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

    logs = []

    if os.path.exists(LOG_FILE):

        try:

            with open(LOG_FILE, "r") as file:
                logs = json.load(file)

        except Exception:
            logs = []

    logs.append(data)

    logs = logs[-100:]

    with open(LOG_FILE, "w") as file:
        json.dump(
            logs,
            file,
            indent=4
        )


# ============================================================
# ALERT FUNCTION
# ============================================================

def show_alert():

    if os.path.exists(ALERT_SOUND):

        try:

            with open(
                ALERT_SOUND,
                "rb"
            ) as audio_file:

                audio_bytes = audio_file.read()

            st.audio(
                audio_bytes,
                format="audio/mp3",
                autoplay=True
            )

        except Exception:
            pass


# ============================================================
# YOLO FRAME DETECTION
# ============================================================

def detect_frame(
    frame,
    confidence=0.30
):

    output_frame = frame.copy()

    weapon_detected = False

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

                weapon_detected = True

                weapon_count += 1

                highest_confidence = max(
                    highest_confidence,
                    conf
                )

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                cv2.rectangle(
                    output_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    3
                )

                label = (
                    f"WEAPON {conf:.2f}"
                )

                cv2.putText(
                    output_frame,
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
        output_frame,
        weapon_detected,
        highest_confidence,
        weapon_count
    )


# ============================================================
# HOME PAGE
# ============================================================

if st.session_state.page == "home":

    st.title("🛡️ WeaponGuard AI")

    st.subheader(
        "AI-Based Weapon Detection Using CCTV"
    )

    st.markdown("---")

    col1, col2 = st.columns(
        [2, 1]
    )

    with col1:

        st.markdown(
            """
            ### 🚨 Intelligent Weapon Detection System

            WeaponGuard AI uses **YOLO AI object detection**
            to detect weapons from:

            - 📷 Images
            - 🎥 CCTV Videos
            - 📹 Live Camera

            The system identifies weapons and displays
            a red bounding box with confidence.
            """
        )

        st.markdown("---")

        st.markdown("### 🎯 Features")

        st.markdown(
            """
            ✅ AI-based weapon detection

            ✅ Image detection

            ✅ CCTV video detection

            ✅ Live camera detection

            ✅ Confidence score

            ✅ Red bounding box

            ✅ Audio alert

            ✅ Detection logging
            """
        )

    with col2:

        st.info(
            f"""
            **Project**

            Artificial Intelligence Career for Women (AICW)

            **College**

            {COLLEGE_NAME}

            **Guide**

            {GUIDE_NAME}
            """
        )

    st.markdown("---")

    if st.button(
        "🚀 Start Detection",
        use_container_width=True,
        type="primary"
    ):

        st.session_state.page = "detection"

        st.rerun()


# ============================================================
# DETECTION PAGE
# ============================================================

elif st.session_state.page == "detection":

    st.title("🛡️ Weapon Detection")

    if st.button("⬅️ Back to Home"):

        st.session_state.page = "home"

        st.rerun()

    st.markdown("---")


    # ========================================================
    # THREE TABS
    # ========================================================

    tab1, tab2, tab3 = st.tabs(
        [
            "📷 Image Detection",
            "🎥 Video Detection",
            "📹 Live Detection"
        ]
    )


    # ========================================================
    # IMAGE DETECTION
    # ========================================================

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

            image = Image.open(
                uploaded_image
            ).convert("RGB")

            image_array = np.array(
                image
            )

            st.image(
                image_array,
                caption="Input Image",
                use_container_width=True
            )

            if st.button(
                "🔍 Detect Weapon",
                use_container_width=True,
                type="primary",
                key="image_detect"
            ):

                image_bgr = cv2.cvtColor(
                    image_array,
                    cv2.COLOR_RGB2BGR
                )

                # Low confidence for image
                results = model(
                    image_bgr,
                    conf=0.01,
                    verbose=False
                )

                output_image = (
                    image_bgr.copy()
                )

                weapon_count = 0

                highest_confidence = 0.0

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

                            weapon_count += 1

                            highest_confidence = max(
                                highest_confidence,
                                confidence
                            )

                            x1, y1, x2, y2 = map(
                                int,
                                box.xyxy[0]
                            )

                            cv2.rectangle(
                                output_image,
                                (x1, y1),
                                (x2, y2),
                                (0, 0, 255),
                                3
                            )

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
                                0.7,
                                (0, 0, 255),
                                2
                            )

                output_rgb = cv2.cvtColor(
                    output_image,
                    cv2.COLOR_BGR2RGB
                )

                st.markdown("---")

                if weapon_count > 0:

                    st.error(
                        f"🚨 WEAPON DETECTED: "
                        f"{weapon_count}"
                    )

                    st.write(
                        f"Confidence: "
                        f"{highest_confidence:.2%}"
                    )

                    st.image(
                        output_rgb,
                        caption="Detection Result",
                        use_container_width=True
                    )

                    save_detection_log(
                        "Image",
                        highest_confidence
                    )

                    show_alert()

                else:

                    st.success(
                        "✅ NO WEAPON DETECTED"
                    )

                    st.image(
                        output_rgb,
                        caption="Detection Result",
                        use_container_width=True
                    )


    # ========================================================
    # VIDEO DETECTION
    # ========================================================

    with tab2:

        st.subheader(
            "🎥 Upload CCTV Video"
        )

        uploaded_video = st.file_uploader(
            "Choose a video",
            type=[
                "mp4",
                "avi",
                "mov",
                "mkv"
            ],
            key="video_upload"
        )

        if uploaded_video is not None:

            video_path = (
                "uploaded_video.mp4"
            )

            with open(
                video_path,
                "wb"
            ) as file:

                file.write(
                    uploaded_video.read()
                )

            st.video(video_path)

            if st.button(
                "🔍 Detect Weapons in Video",
                use_container_width=True,
                type="primary",
                key="video_detect"
            ):

                cap = cv2.VideoCapture(
                    video_path
                )

                if not cap.isOpened():

                    st.error(
                        "❌ Could not open video."
                    )

                else:

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

                    frame_count = int(
                        cap.get(
                            cv2.CAP_PROP_FRAME_COUNT
                        )
                    )

                    output_path = (
                        "weapon_detection_output.mp4"
                    )

                    fourcc = (
                        cv2.VideoWriter_fourcc(
                            *"mp4v"
                        )
                    )

                    out = cv2.VideoWriter(
                        output_path,
                        fourcc,
                        fps,
                        (width, height)
                    )

                    progress = st.progress(
                        0
                    )

                    status = st.empty()

                    consecutive_count = 0

                    detected_any = False

                    max_confidence = 0.0

                    current_frame = 0

                    while True:

                        ret, frame = (
                            cap.read()
                        )

                        if not ret:
                            break

                        current_frame += 1

                        (
                            output_frame,
                            weapon_detected,
                            frame_confidence,
                            weapon_count
                        ) = detect_frame(
                            frame,
                            confidence_threshold
                        )

                        if weapon_detected:

                            consecutive_count += 1

                            max_confidence = max(
                                max_confidence,
                                frame_confidence
                            )

                        else:

                            consecutive_count = 0

                        if (
                            consecutive_count
                            >= required_frames
                        ):

                            detected_any = True

                            cv2.putText(
                                output_frame,
                                "WEAPON DETECTED",
                                (20, 45),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.0,
                                (0, 0, 255),
                                3
                            )

                        out.write(
                            output_frame
                        )

                        if frame_count > 0:

                            progress.progress(
                                min(
                                    current_frame /
                                    frame_count,
                                    1.0
                                )
                            )

                        status.text(
                            f"Processing frame "
                            f"{current_frame}/"
                            f"{frame_count}"
                        )

                    cap.release()

                    out.release()

                    progress.progress(1.0)

                    status.success(
                        "✅ Video processing completed."
                    )

                    st.markdown("---")

                    if detected_any:

                        st.error(
                            "🚨 WEAPON DETECTED "
                            "IN VIDEO"
                        )

                        st.write(
                            f"Highest confidence: "
                            f"{max_confidence:.2%}"
                        )

                        save_detection_log(
                            "Video",
                            max_confidence
                        )

                        show_alert()

                    else:

                        st.success(
                            "✅ NO WEAPON DETECTED "
                            "IN VIDEO"
                        )

                    st.video(
                        output_path
                    )

                    with open(
                        output_path,
                        "rb"
                    ) as file:

                        st.download_button(
                            "⬇️ Download Detection Video",
                            data=file,
                            file_name=(
                                "weapon_detection_output.mp4"
                            ),
                            mime="video/mp4",
                            use_container_width=True
                        )


    # ========================================================
    # LIVE DETECTION
    # ========================================================

    with tab3:

        st.subheader(
            "📹 Live Camera Detection"
        )

        st.info(
            """
            📌 Click START below.

            Your browser will ask for camera permission.
            Select **Allow** to start live detection.
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


        # ====================================================
        # LIVE CAMERA PROCESSOR
        # ====================================================

        class LiveWeaponDetector(
            VideoProcessorBase
        ):

            def __init__(self):

                self.last_log_time = 0

            def recv(self, frame):

                img = frame.to_ndarray(
                    format="bgr24"
                )

                output = img.copy()

                results = model(
                    img,
                    conf=live_confidence,
                    verbose=False
                )

                weapon_found = False

                highest_confidence = 0.0

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

                            weapon_found = True

                            highest_confidence = max(
                                highest_confidence,
                                confidence
                            )

                            x1, y1, x2, y2 = map(
                                int,
                                box.xyxy[0]
                            )

                            cv2.rectangle(
                                output,
                                (x1, y1),
                                (x2, y2),
                                (0, 0, 255),
                                3
                            )

                            label = (
                                f"WEAPON "
                                f"{confidence:.2f}"
                            )

                            cv2.putText(
                                output,
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


                # =================================================
                # STATUS TEXT
                # =================================================

                if weapon_found:

                    cv2.rectangle(
                        output,
                        (0, 0),
                        (
                            output.shape[1],
                            60
                        ),
                        (0, 0, 255),
                        -1
                    )

                    cv2.putText(
                        output,
                        "🚨 WEAPON DETECTED",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        3
                    )

                    current_time = time.time()

                    # Log every 5 seconds
                    if (
                        current_time
                        - self.last_log_time
                        > 5
                    ):

                        save_detection_log(
                            "Live Camera",
                            highest_confidence
                        )

                        self.last_log_time = (
                            current_time
                        )

                else:

                    cv2.putText(
                        output,
                        "NO WEAPON DETECTED",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2
                    )

                return av.VideoFrame.from_ndarray(
                    output,
                    format="bgr24"
                )


        # ====================================================
        # START WEB CAMERA
        # ====================================================

        webrtc_streamer(
            key="weapon-live-detection",
            video_processor_factory=LiveWeaponDetector,
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            async_processing=True
        )

        st.markdown("---")

        st.warning(
            """
            ⚠️ If the camera does not start:

            1. Check browser camera permission.
            2. Make sure no other application is using the camera.
            3. Use Chrome or Edge.
            4. Refresh the page and click START again.
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "WeaponGuard AI | AICW | VSM College of Engineering"
)
