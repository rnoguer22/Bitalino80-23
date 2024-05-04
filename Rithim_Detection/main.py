from src.detect_faces import detect_faces, detect_keypoints
from src.signal_processing import butter_bandpass_filter, obtener_señales_interpoladas, reduccion_dimensionalidad_PCA, calcular_frecuencia_cardiaca

from src.plot import show_realtime_graph
import cv2
import numpy as np

if __name__ == '__main__':

    cap = cv2.VideoCapture(0)

    # Lista para almacenar los valores aproximados del pulso cardíaco
    pulse_values = []

    while True:
        # Capturar un frame de la cámara
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el frame.")
            break

        faces = detect_faces(frame)

        # Detectar los puntos clave en la imagen
        keypoints_in_faces = detect_keypoints(frame, faces)

        # Mostrar los puntos clave en la imagen
        for keypoints in keypoints_in_faces:
            for point in keypoints:
                x, y = point.ravel()
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

        #Simulación de la señal de pulso cardíaco

        pulse_value = np.random.randint(60, 100)  

        #Mostrar las revoluciones por minuto en la pantalla

        for (x, y, w, h) in faces:
            
            texto = f'Pulse: {pulse_value} bpm'
            posicion_texto = (x, y - 10)  
            font = cv2.FONT_HERSHEY_SIMPLEX
            escala = 1
            color = (0, 255, 0)
            grosor = 1
        
            cv2.putText(frame, texto, posicion_texto, font, escala, color, grosor, cv2.LINE_AA)

        pulse_values.append(pulse_value)

       
        result_frame = show_realtime_graph(frame, pulse_values)
        cv2.imshow('Frame', result_frame)

        # Salir del bucle si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la cámara y cerrar todas las ventanas
    cap.release()
    cv2.destroyAllWindows()