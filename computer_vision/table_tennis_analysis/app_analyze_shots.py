# === Importação de bibliotecas ===
import streamlit as st
import os
import cv2
import pickle
import mediapipe as mp
import pandas as pd
import numpy as np
import math
import time

# === Configuração inicial ===
mp_pose = mp.solutions.pose  # Inicializa módulo de pose do MediaPipe
st.set_page_config(layout="wide")  # Layout da página no modo largo
movement = 'forehand'  # Tipo de movimento analisado (ex: forehand de tênis)

# === Inicialização de variáveis na sessão ===
for key in ["idx", "idx2", "is_playing", "angles_hist", "precomputed", "fps_my", "fps_pro"]:
    if key not in st.session_state:
        if key in ["idx", "idx2"]:
            st.session_state[key] = 0  # Índices de frame para cada vídeo
        elif key == "is_playing":
            st.session_state[key] = False  # Flag de reprodução
        else:
            st.session_state[key] = {}  # Dicionários para guardar dados de ângulo, FPS, etc.

# === Funções para carregar dados ===
@st.cache_resource
def load_data(video, video2):
    # Carrega os dados de landmarks (pontos do corpo) pré-computados
    with open(f"{movement}_landmarks/landmarks_{video.split('.')[0]}.pickle", 'rb') as f:
        lands_data = pickle.load(f)
    with open(f"{movement}_landmarks/landmarks_{video2.split('.')[0]}.pickle", 'rb') as f:
        lands_data2 = pickle.load(f)
    return lands_data, lands_data2

@st.cache_resource
def load_videos(video, video2):
    # Abre os vídeos usando OpenCV
    cap_my = cv2.VideoCapture(f'{movement}/{video}')
    cap_pro = cv2.VideoCapture(f'{movement}/{video2}')
    return {"my": cap_my, "pro": cap_pro}

# === Função de cálculo de ângulo entre três pontos ===
def calc_angle(landmark, p1, p2, p3):
    x1, y1 = landmark[p1].x, landmark[p1].y
    x2, y2 = landmark[p2].x, landmark[p2].y
    x3, y3 = landmark[p3].x, landmark[p3].y

    # Calcula os lados do triângulo
    a = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    b = math.sqrt((x2 - x3)**2 + (y2 - y3)**2)
    c = math.sqrt((x3 - x1)**2 + (y3 - y1)**2)

    # Aplica a Lei dos Cossenos para obter o ângulo
    cos_gamma = (a**2 + b**2 - c**2) / (2 * a * b)
    angle = math.degrees(math.acos(np.clip(cos_gamma, -1.0, 1.0)))
    return angle

# === Função para desenhar a "sombra" (trajetória) do movimento ===
def draw_shadow(frame, lands_data, idx, bp, h, w):
    line_points = [(int(i.landmark[bp].x * w), int(i.landmark[bp].y * h)) for i in lands_data[:idx]]
    trajectory_color = (200, 200, 50)
    for i in range(1, len(line_points)):
        thickness = int(np.sqrt(30 / float(i + 1)) * 2.5)
        cv2.line(frame, line_points[i - 1], line_points[i], trajectory_color, thickness)
    return frame

# === Função para anotar o ângulo calculado no vídeo ===
def annotate_angle(frame, angle, landmark, p2, h, w):
    position = (int(landmark[p2].x * w + 10), int(landmark[p2].y * h + 10))
    cv2.putText(frame, str(int(angle)), position, cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2)
    return frame

# === Função que cria os controles na barra lateral ===
def draw_sidebar(landmarks, landmarks2):
    expander_frame = st.sidebar.expander("🎛️ Frame Controller", True)
    slider_play = expander_frame.empty()
    slider_ref_play = expander_frame.empty()
    col3, col4 = expander_frame.columns(2)
    col7, col8 = st.sidebar.columns(2)

    # Controles de navegação manual por frame
    if col3.button('-1'):  # Retroceder vídeo principal
        st.session_state.idx = max(st.session_state.idx - 1, 0)
    if col4.button('+1'):  # Avançar vídeo principal
        st.session_state.idx = min(st.session_state.idx + 1, len(landmarks)-1)
    if col3.button('-1 R'):  # Retroceder vídeo de referência
        st.session_state.idx2 = max(st.session_state.idx2 - 1, 0)
    if col4.button('+1 R'):  # Avançar vídeo de referência
        st.session_state.idx2 = min(st.session_state.idx2 + 1, len(landmarks2)-1)

    # Botão único de Play/Pause
    if col7.button('Play / Pause'):
        st.session_state.is_playing = not st.session_state.is_playing

    # Botão de Reset
    if col8.button("🔄 Reset"):
        st.session_state.idx = 0
        st.session_state.idx2 = 0

    # Sliders sincronizados para navegar nos frames
    st.session_state["idx"] = slider_play.slider('Main Video', 0, len(landmarks) - 1, st.session_state["idx"])
    st.session_state["idx2"] = slider_ref_play.slider('Ref Video', 0, len(landmarks2) - 1, st.session_state["idx2"])

# === Lógica principal do app ===
# Listagem dos vídeos disponíveis
video_files = sorted([i for i in os.listdir(movement) if i[0] != '.'])
video = st.sidebar.selectbox("Selecione um vídeo:", video_files)
video2 = st.sidebar.selectbox("Selecione um vídeo de ref:", video_files, index=len(video_files)-1)

if video and video2:
    # Dicionário de partes do corpo e pontos de referência para ângulo
    angle_parts = {
        "right arm": (12, 14, 16),
        "left arm": (11, 13, 15),
        "right leg": (24, 26, 28),
        "left leg": (23, 25, 27),
        "right_torso": (12, 24, 26),
        "left_torso": (11, 23, 25)
    }

    # Seleção das partes a serem medidas
    ap = st.multiselect("Angle on", angle_parts.keys(), ["right arm"])

    # Carrega landmarks e vídeos
    lands_data, lands_data2 = load_data(video, video2)
    cap = load_videos(video, video2)

    # Desenha controles interativos
    draw_sidebar(lands_data, lands_data2)

    # Salva os FPS dos vídeos
    st.session_state.fps_my = cap["my"].get(cv2.CAP_PROP_FPS)
    st.session_state.fps_pro = cap["pro"].get(cv2.CAP_PROP_FPS)

    # Prepara layout de exibição
    col1, col2 = st.columns(2)
    ph1, ph2 = col1.empty(), col2.empty()
    h, w = int(cap["my"].get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap["my"].get(cv2.CAP_PROP_FRAME_WIDTH))
    h2, w2 = int(cap["pro"].get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap["pro"].get(cv2.CAP_PROP_FRAME_WIDTH))

    # Slider de velocidade de reprodução
    play_speed = st.sidebar.slider("Velocidade do Play (seg)", 0.01, 0.5, 0.1)

    # Pré-computação de ângulos se ainda não feita
    if ap and ap[0] not in st.session_state.precomputed:
        i = ap[0]
        ap_tup = angle_parts[i]
        my_series = [calc_angle(frame.landmark, *ap_tup) for frame in lands_data]
        pro_series = [calc_angle(frame.landmark, *ap_tup) for frame in lands_data2]
        st.session_state.precomputed[i] = {"my": my_series, "pro": pro_series}

    # === Modo parado (visualização estática) ===
    if not st.session_state["is_playing"]:
        cap["my"].set(cv2.CAP_PROP_POS_FRAMES, st.session_state["idx"])
        cap["pro"].set(cv2.CAP_PROP_POS_FRAMES, st.session_state["idx2"])
        ret1, frame1 = cap["my"].read()
        ret2, frame2 = cap["pro"].read()

        if ret1 and ret2:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            for i in ap:
                ap_tup = angle_parts[i]
                frame1 = draw_shadow(frame1, lands_data, st.session_state["idx"], ap_tup[2], h, w)
                frame2 = draw_shadow(frame2, lands_data2, st.session_state["idx2"], ap_tup[2], h2, w2)
                idx1_safe = min(st.session_state["idx"], len(st.session_state.precomputed[i]["my"]) - 1)
                idx2_safe = min(st.session_state["idx2"], len(st.session_state.precomputed[i]["pro"]) - 1)
                a1 = st.session_state.precomputed[i]["my"][idx1_safe]
                a2 = st.session_state.precomputed[i]["pro"][idx2_safe]
                frame1 = annotate_angle(frame1, a1, lands_data[idx1_safe].landmark, ap_tup[1], h, w)
                frame2 = annotate_angle(frame2, a2, lands_data2[idx2_safe].landmark, ap_tup[1], h2, w2)
            ph1.image(frame1, channels="RGB", use_container_width=True)
            ph2.image(frame2, channels="RGB", use_container_width=True)

    # === Modo de reprodução automática ===
    while st.session_state["is_playing"]:
        if st.session_state["idx"] < len(lands_data)-1:
            st.session_state["idx"] += 1

        # Sincroniza segundo vídeo com base no tempo do primeiro
        time_sec = st.session_state["idx"] / st.session_state.fps_my
        idx2_sync = int(time_sec * st.session_state.fps_pro)
        st.session_state["idx2"] = min(idx2_sync, len(lands_data2)-1)

        # Lê os frames sincronizados
        cap["my"].set(cv2.CAP_PROP_POS_FRAMES, st.session_state["idx"])
        cap["pro"].set(cv2.CAP_PROP_POS_FRAMES, st.session_state["idx2"])
        ret1, frame1 = cap["my"].read()
        ret2, frame2 = cap["pro"].read()
        
        if not ret1 or not ret2:
            st.session_state["is_playing"] = False
            break

        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        for i in ap:
            ap_tup = angle_parts[i]
            frame1 = draw_shadow(frame1, lands_data, st.session_state["idx"], ap_tup[2], h, w)
            frame2 = draw_shadow(frame2, lands_data2, st.session_state["idx2"], ap_tup[2], h2, w2)
            idx1_safe = min(st.session_state["idx"], len(st.session_state.precomputed[i]["my"]) - 1)
            idx2_safe = min(st.session_state["idx2"], len(st.session_state.precomputed[i]["pro"]) - 1)
            a1 = st.session_state.precomputed[i]["my"][idx1_safe]
            a2 = st.session_state.precomputed[i]["pro"][idx2_safe]
            frame1 = annotate_angle(frame1, a1, lands_data[idx1_safe].landmark, ap_tup[1], h, w)
            frame2 = annotate_angle(frame2, a2, lands_data2[idx2_safe].landmark, ap_tup[1], h2, w2)
        ph1.image(frame1, channels="RGB", use_container_width=True)
        ph2.image(frame2, channels="RGB", use_container_width=True)
        time.sleep(play_speed)
