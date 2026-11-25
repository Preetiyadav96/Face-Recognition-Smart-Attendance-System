
import streamlit as st
import sqlite3
import pandas as pd
import os
import cv2
import av
import json
from streamlit_webrtc import webrtc_streamer
from attendance import mark_attendance


# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="FaceAI | Smart Attendance",
    page_icon="👤",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "lbph_model.yml")
LABELS_PATH = os.path.join(BASE_DIR, "model", "labels.json")
DATABASE_PATH = os.path.join(BASE_DIR, "database", "attendance.db")
DATASET_PATH = os.path.join(BASE_DIR, "dataset")


# ---------------- PROFESSIONAL CSS ----------------

st.markdown("""
<style>
.stApp {
    background-color: #f3f6fb;
}

[data-testid="stSidebar"] {
    background-color: #101c36;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.main-title {
    font-size: 32px;
    font-weight: 800;
    color: #172b4d;
}

.subtitle {
    color: #718096;
    font-size: 15px;
}

div[data-testid="stMetric"] {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e5eaf2;
}
</style>
""", unsafe_allow_html=True)


# ---------------- FACE DETECTION MODEL ----------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ---------------- FACE RECOGNITION MODEL ----------------

recognizer = cv2.face.LBPHFaceRecognizer_create()

model_loaded = False
label_names = {}
model_error = ""

try:
    recognizer.read(MODEL_PATH)

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        label_names = json.load(file)

    model_loaded = True

except Exception as e:
    model_error = str(e)


# ---------------- LIVE FACE RECOGNITION ----------------

marked_today = set()
def video_callback(frame):

    img = frame.to_ndarray(format="bgr24")

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        try:
            label, confidence = recognizer.predict(face)

            if confidence < 70:

                employee = label_names.get(
                    str(label),
                    "Unknown"
                )

                if "_" in employee:
                    employee_id, employee_name = employee.split("_", 1)
                else:
                    employee_id = "N/A"
                    employee_name = employee

                display_text = f"{employee_name} | ID: {employee_id}"
                color = (0, 255, 0)

            else:
                display_text = "Unknown Person"
                color = (0, 0, 255)

        except Exception:
            display_text = "Recognition Error"
            color = (0, 0, 255)

        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        cv2.putText(
            img,
            display_text,
            (x, max(y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    cv2.putText(
        img,
        f"Faces Detected: {len(faces)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    return av.VideoFrame.from_ndarray(
        img,
        format="bgr24"
    )


# ---------------- ATTENDANCE DATABASE ----------------

def get_attendance():

    empty_df = pd.DataFrame(
        columns=["id", "name", "date", "time", "status"]
    )

    if not os.path.exists(DATABASE_PATH):
        return empty_df

    try:
        conn = sqlite3.connect(DATABASE_PATH)

        df = pd.read_sql_query(
            "SELECT * FROM attendance",
            conn
        )

        conn.close()

        if "id" in df.columns:
            df = df.sort_values(
                by="id",
                ascending=False
            )

        return df

    except Exception:
        return empty_df


# ---------------- REGISTERED EMPLOYEES ----------------

def get_registered_employees():

    if not os.path.exists(DATASET_PATH):
        return 0

    return len([
        folder for folder in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, folder))
    ])


df = get_attendance()
registered_employees = get_registered_employees()


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.markdown("# 👤 FaceAI")

    st.caption("SMART RECOGNITION SYSTEM")

    st.divider()

    page = st.radio(
        "MAIN MENU",
        [
            "Dashboard",
            "Live Face Detection",
            "Attendance Records",
            "About Project"
        ],
        key="main_navigation"
    )

    st.divider()

    st.caption("AI Face Recognition System")
    st.caption("Version 1.0")


# ---------------- DASHBOARD ----------------

if page == "Dashboard":

    st.markdown(
        '<div class="main-title">Dashboard Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">AI-powered face recognition and attendance management</div>',
        unsafe_allow_html=True
    )

    total_records = len(df)

    total_people = registered_employees

    today = pd.Timestamp.now().strftime("%Y-%m-%d")

    if not df.empty and "date" in df.columns:
        today_records = len(
            df[df["date"].astype(str) == today]
        )
    else:
        today_records = 0

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Attendance",
        total_records
    )

    c2.metric(
        "Registered Employees",
        total_people
    )

    c3.metric(
        "Today's Attendance",
        today_records
    )

    st.subheader("Daily Attendance Analytics")

    if not df.empty and "date" in df.columns:

        daily_data = (
            df.groupby("date")
            .size()
            .reset_index(name="Total Present")
        )

        daily_data = daily_data.sort_values("date")

        st.bar_chart(
            daily_data.set_index("date")
        )

    else:
        st.info("No attendance data available for chart.")

    st.subheader("Recent Attendance")

    if not df.empty:

        st.dataframe(
            df.head(10),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info("No attendance records available yet.")

    st.subheader("System Information")

    a, b = st.columns(2)

    a.metric("Face Detection Engine", "OpenCV")
    b.metric("Database", "SQLite")


# ---------------- LIVE FACE RECOGNITION ----------------

elif page == "Live Face Detection":

    st.markdown(
        '<div class="main-title">Live Face Recognition</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Identify registered employees using AI</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Start your webcam to recognize registered faces."
    )

    st.info(
        "Allow camera permission when your browser asks for it."
    )

    if face_cascade.empty():

        st.error("Face detection model could not be loaded.")

    elif not model_loaded:

        st.error(
            "Face recognition model is not available. "
            "Please train the model first."
        )

        st.caption(model_error)

    else:

        webrtc_streamer(
            key="face-recognition",
            video_frame_callback=video_callback,
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            async_processing=True
        )


# ---------------- ATTENDANCE RECORDS ----------------

elif page == "Attendance Records":

    st.markdown(
        '<div class="main-title">Attendance Management</div>',
        unsafe_allow_html=True
    )

    if not df.empty:

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        csv = df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Attendance CSV",
            data=csv,
            file_name="attendance_report.csv",
            mime="text/csv"
        )

    else:

        st.info("No attendance data found.")


# ---------------- ABOUT PROJECT ----------------

elif page == "About Project":

    st.markdown(
        '<div class="main-title">About FaceAI</div>',
        unsafe_allow_html=True
    )

    st.write("""
    FaceAI is an AI-based Face Detection and Recognition System.

    ### Technologies

    - Python
    - OpenCV
    - LBPH Face Recognizer
    - NumPy
    - Pandas
    - Streamlit
    - SQLite

    ### Implemented Features

    - Employee face registration
    - Face image dataset creation
    - LBPH model training
    - Real-time webcam integration
    - Face recognition
    - Unknown person identification
    - Attendance records
    - CSV report download

    ### Upcoming Features

    - Automatic attendance marking
    - Attendance analytics
    - Employee management
    """)
