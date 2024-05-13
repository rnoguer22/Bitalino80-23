def calculate_bounding_box(keypoints):
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')
    
    # Iterar sobre los keypoints para encontrar los valores mínimos y máximos
    for point in keypoints:
        x, y = point
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
    
    # Calcular el ancho y la altura del rectángulo delimitador
    width = max_x - min_x
    height = max_y - min_y
    
    # Devolver las coordenadas del rectángulo delimitador
    return min_x, min_y, width, height