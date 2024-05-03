import cv2
import numpy as np
from src.detect_faces import detect_faces, detect_keypoints, select_rois


def extract_ppg_signal(frame, roi):
    frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    roi_y = frame_yuv[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2], 0]
    ppg_signal = np.mean(roi_y, axis=(0, 1))
    return ppg_signal

# Función para procesar un frame y extraer la señal PPG de las caras detectadas
def process_frame(frame):
    # Detectar caras en el frame
    faces = detect_faces(frame)
    
    # Seleccionar ROIs en las caras detectadas
    rois = select_rois(frame, faces)
    
    ppg_signals = []
    for roi in rois:
        # Detectar puntos clave en la ROI
        keypoints = detect_keypoints(frame, [roi])
        
        # Extraer la señal PPG de la ROI que contiene los puntos clave
        ppg_signal = extract_ppg_signal(frame, roi)
        
        # Agregar la señal PPG a la lista de señales
        ppg_signals.append(ppg_signal)
    
    return ppg_signals

