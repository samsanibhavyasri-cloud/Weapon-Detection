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
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    st.error("❌ best.pt not found!")
    st.info(
        "Please keep best.pt in the same folder as app.py."
    )
    st.stop()


try:
    model = load_model()

except Exception as e:
    st.error(f"❌ Could not load YOLO model: {e}")
    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "home"


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

    # Keep latest 100 detections
    logs = logs[-100:]

    with open(LOG_FILE, "w") as file:
        json.dump(
            logs,
            file,
            indent=4
        )


# ============================================================
# COMMON YOLO DETECTION FUNCTION
# ============================================================

def detect_frame(frame, confidence=0.30):

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

            conf = float(box.conf[0])
            class_id = int(box.cls[0])

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

                label = f"WEAPON {conf:.2f}"

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

    # --------------------------------------------------------
    # MAIN PAGE
    # --------------------------------------------------------

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

            WeaponGuard AI uses **YOLO object detection**
            to identify weapons from:

            - 📷 Images
            - 🎥 CCTV Videos
            - 📹 Live Camera

            When a weapon is detected, the system
            displays a red bounding box and confidence.
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

            ✅ Red weapon bounding box

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
# IMAGE DETECTION
# ============================================================

def image_detection():

    st.subheader("📷 Upload Image")

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

    if uploaded_image is None:
        return

    image = Image.open(
        uploaded_image
    ).convert("RGB")

    image_array = np.array(image)

    st.image(
        image_array,
        caption="Input Image",
        use_container_width=True
    )

    if st.button(
        "🔍 Detect Weapon",
        use_container_width=True,
        type="primary",
        key="image_detect_button"
    ):

        image_bgr = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2BGR
        )

        # Low threshold for image detection
        results = model(
            image_bgr,
            conf=0.01,
            verbose=False
        )

        output = image_bgr.copy()

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

                if class_name.lower() == "weapon":

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
                        output,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        3
                    )

                    cv2.putText(
                        output,
                        f"WEAPON {confidence:.2f}",
                        (
                            x1,
                            max(y1 - 10, 30)
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

        output_rgb = cv2.cvtColor(
            output,
            cv2.COLOR_BGR2RGB
        )

        st.markdown("---")

        if weapon_count > 0:

            st.error(
                f"🚨 WEAPON DETECTED: {weapon_count}"
            )

            st.write(
                f"Highest confidence: "
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

        else:

            st.success(
                "✅ NO WEAPON DETECTED"
            )

            st.image(
                output_rgb,
                caption="Detection Result",
                use_container_width=True
            )


# ============================================================
# VIDEO DETECTION
# ============================================================

def video_detection(
    confidence_threshold,
    required_frames
):

    st.subheader("🎥 Upload CCTV Video")

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

    if uploaded_video is None:
        return

    video_path = "uploaded_video.mp4"

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
        key="video_detect_button"
    ):

        cap = cv2.VideoCapture(
            video_path
        )

        if not cap.isOpened():

            st.error(
                "❌ Could not open video."
            )

            return

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

        output_path = (
            "weapon_detection_output.mp4"
        )

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        progress = st.progress(0)

        status = st.empty()

        current_frame = 0
        consecutive_count = 0

        detected_any = False
        max_confidence = 0.0

        while True:

            ret, frame = cap.read()

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
                    "🚨 WEAPON DETECTED",
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3
                )

            out.write(
                output_frame
            )

            if total_frames > 0:

                progress.progress(
                    min(
                        current_frame /
                        total_frames,
                        1.0
                    )
                )

            status.text(
                f"Processing frame "
                f"{current_frame}/{total_frames}"
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
                "🚨 WEAPON DETECTED IN VIDEO"
            )

            st.write(
                f"Highest confidence: "
                f"{max_confidence:.2%}"
            )

            save_detection_log(
                "Video",
                max_confidence
            )

        else:

            st.success(
                "✅ NO WEAPON DETECTED IN VIDEO"
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


# ============================================================
# OPEN CV LIVE CAMERA DETECTION
# ============================================================

def live_detection(
    confidence_threshold
):

    st.subheader(
        "📹 Live Camera Detection"
    )

    st.info(
        """
        📌 This version uses your computer's webcam
        directly through OpenCV.

        Click **START LIVE DETECTION**.

        A separate camera window will open.

        Press **Q** inside the camera window
        to stop detection.
        """
    )

    live_confidence = st.slider(
        "Live Detection Confidence",
        min_value=0.05,
        max_value=0.95,
        value=0.20,
        step=0.05,
        key="opencv_live_confidence"
    )

    st.warning(
        "⚠️ Make sure no other application is using your webcam."
    )

    if st.button(
        "📹 START LIVE DETECTION",
        type="primary",
        use_container_width=True,
        key="start_opencv_live"
    ):

        # ----------------------------------------------------
        # OPEN CAMERA
        # ----------------------------------------------------

        camera = cv2.VideoCapture(
            0,
            cv2.CAP_DSHOW
        )

        # If camera 0 fails, try camera 1
        if not camera.isOpened():

            camera.release()

            camera = cv2.VideoCapture(
                1,
                cv2.CAP_DSHOW
            )

        if not camera.isOpened():

            st.error(
                """
                ❌ Could not open your webcam.

                Please check:
                • Camera permission
                • Camera connection
                • Windows Camera app
                • Another application using the camera
                """
            )

            return

        # ----------------------------------------------------
        # CAMERA SETTINGS
        # ----------------------------------------------------

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
            "🟢 Camera started! Look at the separate camera window."
        )

        st.info(
            "Press Q on the camera window to stop."
        )

        # ----------------------------------------------------
        # LIVE LOOP
        # ----------------------------------------------------

        last_log_time = 0

        frame_counter = 0

        while True:

            success, frame = camera.read()

            if not success:

                st.error(
                    "❌ Could not read camera frame."
                )

                break

            frame_counter += 1

            # ------------------------------------------------
            # YOLO DETECTION
            # ------------------------------------------------

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
            # STATUS
            # ------------------------------------------------

            if weapon_found:

                # Red top banner
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

                # Log once every 5 seconds
                current_time = time.time()

                if (
                    current_time
                    - last_log_time
                    >= 5
                ):

                    save_detection_log(
                        "Live Camera",
                        highest_confidence
                    )

                    last_log_time = current_time

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

            # ------------------------------------------------
            # PRESS Q TO EXIT
            # ------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):

                break

            # ESC also stops
            if key == 27:

                break

        # ----------------------------------------------------
        # RELEASE CAMERA
        # ----------------------------------------------------

        camera.release()

        cv2.destroyAllWindows()

        # Give Windows time to close the window
        for _ in range(3):
            cv2.waitKey(1)

        st.success(
            "🟢 Live Detection stopped."
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
            "## ⚙️ Detection Settings"
        )

        st.markdown("---")

        if st.button(
            "⬅️ Back to Home",
            use_container_width=True
        ):

            st.session_state.page = "home"

            st.rerun()

        st.markdown("---")

        confidence_threshold = st.slider(
            "🎯 Confidence Threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.30,
            step=0.05
        )

        required_frames = st.slider(
            "🎞️ Required Consecutive Frames",
            min_value=1,
            max_value=10,
            value=3,
            step=1
        )

        st.markdown("---")

        st.write("🤖 **Model:** YOLO")

        st.write("🎯 **Detection:** Weapon")

        st.write(
            "📷 **Input:** Image"
        )

        st.write(
            "🎥 **Input:** CCTV Video"
        )

        st.write(
            "📹 **Input:** Live Camera"
        )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.title(
        "🚨 WeaponGuard AI"
    )

    st.caption(
        "AI-powered weapon detection using YOLO"
    )

    st.markdown("---")

    # --------------------------------------------------------
    # THREE TABS
    # --------------------------------------------------------

    tab1, tab2, tab3 = st.tabs(
        [
            "📷 Image Detection",
            "🎥 Video Detection",
            "📹 Live Detection"
        ]
    )

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    with tab1:

        image_detection()

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    with tab2:

        video_detection(
            confidence_threshold,
            required_frames
        )

    # --------------------------------------------------------
    # LIVE
    # --------------------------------------------------------

    with tab3:

        live_detection(
            confidence_threshold
        )


# ============================================================
# MAIN APPLICATION
# ============================================================

if st.session_state.page == "home":

    home_page()

else:

    detection_page()


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "WeaponGuard AI | Artificial Intelligence Career for Women (AICW) | "
    "VSM College of Engineering"
)
