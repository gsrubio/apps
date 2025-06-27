import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
from collections import deque
import tempfile

# Inicialização do MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

movement = 'forehand'
# Diretórios de saída
base_dir = os.path.dirname(os.path.abspath(__file__))
forehand_dir = os.path.join(base_dir, movement)
land_dir = os.path.join(base_dir, f'{movement}_landmarks')

os.makedirs(forehand_dir, exist_ok=True)
os.makedirs(land_dir, exist_ok=True)

# Interface Streamlit
st.title("Forehand Drive Detector - Table Tennis")

uploaded_file = st.file_uploader("Upload seu vídeo de treino (MP4)", type=['mp4'])
scale_percent = st.slider("Select Scale Percentage", min_value=10, max_value=100, value=100)

if uploaded_file is not None:
    st.success("Arquivo recebido! Iniciando processamento...")

    # Salva o arquivo temporariamente
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    filename = uploaded_file.name
    filepath = tfile.name

    # Inicializa captura de vídeo
    cap = cv2.VideoCapture(filepath)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    min_visibility = 0.8

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_to_start = int(total_frames * 0.01)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_start)

    frames = deque(maxlen=fps * 1)
    landmarks_frames = deque(maxlen=fps * 1)

    condition1_met = condition2_met = False
    forehand_count = 0
    post_condition2_frames = 0
    current_frame = 0

    progress_bar = st.progress(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensionamento para melhorar performance
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        current_frame += 1
        completion_percentage = (current_frame / total_frames)
        progress_bar.progress(min(completion_percentage, 1.0))

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Desenha landmarks no frame para visualização
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

            landmarks_frames.append(results.pose_landmarks)
            frames.append(frame)

            wrist_visible = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].visibility > min_visibility
            elbow_visible = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].visibility > min_visibility
            shoulder_visible = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility > min_visibility

            if wrist_visible:
                wrist_back = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x < landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x
                if wrist_back:
                    condition1_met = True

            if condition1_met and wrist_visible and elbow_visible and shoulder_visible:
                wrist_forward = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x > landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x
                if wrist_forward:
                    condition2_met = True

            if condition2_met:
                post_condition2_frames += 1

            if condition1_met and condition2_met and out is None:
                h, w, _ = frame.shape
                forehand_count += 1
                out = cv2.VideoWriter(
                    os.path.join(forehand_dir, f'{filename.split(".")[0]}_{forehand_count}.mp4'),
                    fourcc, fps, (w, h)
                )

            if out is not None:
                if post_condition2_frames >= fps * 0.3:
                    with open(os.path.join(land_dir, f'landmarks_{filename.split(".")[0]}_{forehand_count}.pickle'), 'wb') as f:
                        pickle.dump(list(landmarks_frames), f)

                    while frames:
                        out.write(frames.popleft())

                    out.release()
                    out = None
                    st.write(f"Forehand Drive {forehand_count} salvo")

                    condition1_met = condition2_met = False
                    post_condition2_frames = 0

    cap.release()
    st.success("Processamento concluído!")
