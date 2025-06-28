import cv2
import streamlit as st
from ultralytics import YOLO
import tempfile
import numpy as np


model = YOLO("yolov8n.pt")
VEHICLE_CLASSES = [2, 3, 5, 7]  # Автомобили, мотоциклы, автобусы, грузовики

st.title("Детектор транспорта у шлагбаума 🚗")


uploaded_file = st.file_uploader("Загрузите видео", type=["mp4", "avi"])

if uploaded_file is not None:
    # Сохраняем видео во временный файл
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]

        line_y = int(height * 0.33)
        cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)

        results = model.predict(frame, classes=VEHICLE_CLASSES, verbose=False)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if y2 >= line_y:
                    color = (0, 0, 255)
                    cv2.putText(frame, "AT BARRIER", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
                else:
                    color = (0, 255, 0)


                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        st_frame.image(frame, channels="BGR", use_container_width=True)

    cap.release()