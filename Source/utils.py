import os
import cv2
import numpy as np
import shutil
from datetime import datetime

def extract_frames_and_faces(video_path, output_dir, fps=3):
    faces_dir = os.path.join(output_dir, "faces")
    os.makedirs(faces_dir, exist_ok=True)
    video = cv2.VideoCapture(video_path)
    video_fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps) if video_fps > 0 else 1
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    frame_count, saved_count = 0, 0

    while True:
        ret, frame = video.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for i, (x, y, w, h) in enumerate(faces):
                padding = int(w * 0.2)
                x_pad = max(0, x - padding)
                y_pad = max(0, y - padding)
                w_pad = min(frame.shape[1] - x_pad, w + 2 * padding)
                h_pad = min(frame.shape[0] - y_pad, h + 2 * padding)
                face_roi = frame[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
                face_filename = os.path.join(faces_dir, f"face_{saved_count:04d}_{i:02d}.jpg")
                cv2.imwrite(face_filename, face_roi)
            saved_count += 1
        frame_count += 1
    video.release()
    return faces_dir

def predict_image(image_path, model, img_size=(224, 224)):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    class_idx = np.argmax(prediction[0])
    return "Fake" if class_idx == 0 else "Real"

def process_video_and_predict(video_path, model):
    temp_dir = "temp_" + datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs(temp_dir, exist_ok=True)

    face_dir = extract_frames_and_faces(video_path, temp_dir)
    face_images = os.listdir(face_dir)

    results = []
    for img_name in face_images:
        img_path = os.path.join(face_dir, img_name)
        result = predict_image(img_path, model)
        if result:
            results.append(result)

    fake_ratio = results.count("Fake") / len(results) if results else 0
    video_class = "FAKE" if fake_ratio > 0.05 else "REAL"

    # Xoá thư mục tạm
    shutil.rmtree(temp_dir)

    return {
        "total_faces": len(results),
        "fake_count": results.count("Fake"),
        "fake_ratio": round(fake_ratio, 3),
        "prediction": video_class
    }