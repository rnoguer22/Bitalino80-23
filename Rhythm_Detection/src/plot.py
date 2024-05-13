import cv2
import numpy as np
import random

def show_realtime_graph(frame, pulse_values):
    graph_height = frame.shape[0]  # Usar la altura del frame de la cámara
    graph_width = 500
    padding = 20
    x_spacing = 10  # Espaciado entre los puntos del eje x

    # Crear una imagen en negro para la gráfica
    graph = np.zeros((graph_height, graph_width, 3), np.uint8)

    if len(set(pulse_values)) == 1:  # Si todos los elementos son iguales
        max_y = pulse_values[0] + 1
        min_y = pulse_values[0] - 1
    else:
        max_y = max(pulse_values)
        min_y = min(pulse_values)

    # Calcular el desplazamiento necesario para ajustar los puntos del eje x
    shift = max(0, (len(pulse_values) * x_spacing + 2 * padding) - graph_width)

    # Dibujar el eje x con espaciado aumentado y ajuste de desplazamiento
    for i in range(padding - shift, graph_width - padding - shift + 1, x_spacing):
        cv2.line(graph, (i, graph_height - padding), (i, graph_height - padding + 5), (255, 255, 255), 2)

    # Dibujar el eje y
    cv2.line(graph, (padding - shift, padding), (padding - shift, graph_height - padding), (255, 255, 255), 2)

    # Dibujar la gráfica del pulso cardíaco
    if len(pulse_values) > 1:
        for i in range(1, len(pulse_values)):
            y1 = int(graph_height - padding - (pulse_values[i - 1] - min_y) / (max_y - min_y) * (graph_height - 2 * padding))
            y2 = int(graph_height - padding - (pulse_values[i] - min_y) / (max_y - min_y) * (graph_height - 2 * padding))
            cv2.line(graph, (padding - shift + (i - 1) * x_spacing, y1), (padding - shift + i * x_spacing, y2), (0, 0, 255), 2)

    # Combinar la imagen de la gráfica con el frame de la cámara
    result = np.hstack((frame, graph))

    return result

def generar_frecuencia_cardiaca(frecuencia_cardiaca_promedio, i, constant = False):
    # Simular una señal de pulso cardíaco con ruido aleatorio
    if constant == False:
        noise = random.uniform(-2, 2)
        pulse_rate = frecuencia_cardiaca_promedio + noise

        if i % 10 == 0:
            #simular picos
            pulse_rate = frecuencia_cardiaca_promedio + random.randint(10, 20)

        return round(pulse_rate)
    else:
        return frecuencia_cardiaca_promedio
    



   