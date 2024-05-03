import cv2
import numpy as np

def detect_faces(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return faces


def select_rois(frame, faces):
    rois = []
    for (x, y, w, h) in faces:
        # Definir las coordenadas para la región de las mejillas y el labio superior
        cheeks_and_upperlip_y = y + int(h * 0.5)  # Tomamos el 50% de la altura del rostro
        cheeks_and_upperlip_h = int(h * 0.5)  # Tomamos el 50% de la altura del rostro
        cheeks_and_upperlip_roi = (x, cheeks_and_upperlip_y, w, cheeks_and_upperlip_h)

        rois.append(cheeks_and_upperlip_roi)

    return rois

def detect_keypoints(image, faces, max_corners=100, min_distance=10):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints = []

    for (x, y, w, h) in faces:

        roi = gray[y:y+h, x:x+w]

        corners = cv2.goodFeaturesToTrack(roi, maxCorners=max_corners, qualityLevel=0.01, minDistance=min_distance)
        # Ajustar las coordenadas de las esquinas a la posición original en la imagen completa
        if corners is not None:
            corners += np.array([x, y])  # Ajustar las coordenadas al ROI original
            keypoints.append(corners)
    return keypoints


