# === Importação de bibliotecas ===
import streamlit as st
import os
import cv2
import time
import tempfile
import io

# === Configuração inicial ===
st.set_page_config(layout="wide")  # Layout da página no modo largo

# === Inicialização de variáveis na sessão ===
for key in ["idx", "idx2", "is_playing", "fps_my", "fps_pro"]:
    if key not in st.session_state:
        if key in ["idx", "idx2"]:
            st.session_state[key] = 0  # Índices de frame para cada vídeo
        elif key == "is_playing":
            st.session_state[key] = False  # Flag de reprodução
        else:
            st.session_state[key] = {}  # Dicionários para guardar dados de ângulo, FPS, etc.


@st.cache_resource
def load_videos_from_upload(video_file, video_file2):
    # Salva os vídeos temporariamente e abre com OpenCV
    temp1 = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp1.write(video_file.read())
    temp1.flush()
    temp2 = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp2.write(video_file2.read())
    temp2.flush()
    cap_my = cv2.VideoCapture(temp1.name)
    cap_pro = cv2.VideoCapture(temp2.name)
    return {"my": cap_my, "pro": cap_pro, "temp1": temp1, "temp2": temp2}

# === Função para aplicar zoom/crop ===
def apply_zoom_crop(frame, zoom_percentage):
    if zoom_percentage >= 100:
        return frame
    
    h, w = frame.shape[:2]
    crop_amount = (100 - zoom_percentage) / 100.0
    
    # Calculate crop boundaries
    crop_x = int(w * crop_amount / 2)
    crop_y = int(h * crop_amount / 2)
    
    # Crop the frame from all sides
    cropped = frame[crop_y:h-crop_y, crop_x:w-crop_x]
    return cropped

# === Função que cria os controles na barra lateral ===
def draw_sidebar(total_frames1, total_frames2):
    expander_frame = st.sidebar.expander("🎛️ Frame Controller", True)
    col3, col4 = expander_frame.columns(2)
    col7, col8 = st.sidebar.columns(2)

    # Controles de navegação manual por frame
    if col3.button('⏮ Video 1'):  # Retroceder vídeo principal
        st.session_state.idx = max(st.session_state.idx - 1, 0)
    if col4.button('⏭ Video 1'):  # Avançar vídeo principal
        st.session_state.idx = min(st.session_state.idx + 1, total_frames1-1)
    if col3.button('⏮ Video 2'):  # Retroceder vídeo de referência
        st.session_state.idx2 = max(st.session_state.idx2 - 1, 0)
    if col4.button('⏭ Video 2'):  # Avançar vídeo de referência
        st.session_state.idx2 = min(st.session_state.idx2 + 1, total_frames2-1)
    if col3.button('⏪ Ambos'):
        st.session_state.is_playing = False
        st.session_state.idx = max(st.session_state.idx - 1, 0)
        st.session_state.idx2 = max(st.session_state.idx2 - 1, 0)
    if col4.button('⏩ Ambos'):
        st.session_state.is_playing = False
        st.session_state.idx = min(st.session_state.idx + 1, total_frames1-1)
        st.session_state.idx2 = min(st.session_state.idx2 + 1, total_frames2-1)

    # Botão único de Play/Pause
    if col7.button('⏯️ Play/Pause'):
        st.session_state.is_playing = not st.session_state.is_playing

    # Botão de Reset
    if col8.button("🔄 Reset"):
        st.session_state.idx = 0
        st.session_state.idx2 = 0

# === Lógica principal do app ===
# Uploaders for videos
with st.sidebar.expander("📁 Upload Files", expanded=True):
    video_file = st.file_uploader("Upload Main Video (.mp4)", type=["mp4"], key="main_video")
    video_file2 = st.file_uploader("Upload Ref Video (.mp4)", type=["mp4"], key="ref_video")

with st.sidebar.expander("🎛️ Video Settings", expanded=False):
    max_video_height = st.slider("Altura máxima dos vídeos (px)", 100, 1000, 600)
    zoom_percentage1 = st.slider("Zoom (%) Video 1", 50, 100, 100, help="100% = full frame, lower values crop from all sides")
    zoom_percentage2 = st.slider("Zoom (%) Video 2", 50, 100, 100, help="100% = full frame, lower values crop from all sides")

# Check if videos are uploaded
if not (video_file and video_file2):
    st.warning("Por favor, faça upload dos vídeos para continuar.")
    st.stop()

if video_file and video_file2:
    # Carrega vídeos dos uploads
    cap = load_videos_from_upload(video_file, video_file2)

    # Get total frames for each video
    total_frames1 = int(cap["my"].get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames2 = int(cap["pro"].get(cv2.CAP_PROP_FRAME_COUNT))

    # Salva os FPS dos vídeos
    st.session_state.fps_my = cap["my"].get(cv2.CAP_PROP_FPS)
    st.session_state.fps_pro = cap["pro"].get(cv2.CAP_PROP_FPS)

    # Desenha controles interativos
    draw_sidebar(total_frames1, total_frames2)

    # Prepara layout de exibição
    ph1 = st.empty()  # Container for first video
    ph2 = st.empty()  # Container for second video

    # Slider de velocidade de reprodução
    play_speed = st.sidebar.slider("Velocidade do Play (seg)", 0.01, 0.5, 0.1)

    # === Modo parado (visualização estática) ===
    if not st.session_state["is_playing"]:
        cap["my"].set(cv2.CAP_PROP_POS_FRAMES, st.session_state["idx"])
        cap["pro"].set(cv2.CAP_PROP_POS_FRAMES, st.session_state["idx2"])
        ret1, frame1 = cap["my"].read()
        ret2, frame2 = cap["pro"].read()

        if ret1 and ret2:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            
            # Apply zoom/crop effect
            frame1 = apply_zoom_crop(frame1, zoom_percentage1)
            frame2 = apply_zoom_crop(frame2, zoom_percentage2)
            
            # Resize frames to match max height while maintaining aspect ratio
            h1, w1 = frame1.shape[:2]
            h2, w2 = frame2.shape[:2]
            scale1 = max_video_height / h1
            scale2 = max_video_height / h2
            new_size1 = (int(w1 * scale1), max_video_height)
            new_size2 = (int(w2 * scale2), max_video_height)
            frame1 = cv2.resize(frame1, new_size1)
            frame2 = cv2.resize(frame2, new_size2)
            
            ph1.image(frame1, channels="RGB")
            ph2.image(frame2, channels="RGB")

    # === Modo de reprodução automática ===
    # Calculate frame steps based on FPS
    step_my = max(1, round(st.session_state.fps_my / min(st.session_state.fps_my, st.session_state.fps_pro)))
    step_pro = max(1, round(st.session_state.fps_pro / min(st.session_state.fps_my, st.session_state.fps_pro)))
    
    while st.session_state["is_playing"]:
        # Advance frames, clamping to the last frame
        st.session_state["idx"] = min(st.session_state["idx"] + step_my, total_frames1-1)
        st.session_state["idx2"] = min(st.session_state["idx2"] + step_pro, total_frames2-1)

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
        
        # Apply zoom/crop effect
        frame1 = apply_zoom_crop(frame1, zoom_percentage1)
        frame2 = apply_zoom_crop(frame2, zoom_percentage2)
        
        # Resize frames to match max height while maintaining aspect ratio
        h1, w1 = frame1.shape[:2]
        scale1 = max_video_height / h1
        new_size1 = (int(w1 * scale1), max_video_height)
        frame1 = cv2.resize(frame1, new_size1)
        
        h2, w2 = frame2.shape[:2]
        scale2 = max_video_height / h2
        new_size2 = (int(w2 * scale2), max_video_height)
        frame2 = cv2.resize(frame2, new_size2)
        
        ph1.image(frame1, channels="RGB")
        ph2.image(frame2, channels="RGB")
        time.sleep(play_speed)
