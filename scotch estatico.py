from ultralytics import YOLO
import cv2
import torch

# Verificar disponibilidad de CUDA
#print(torch.cuda.is_available())
#print("procesador:", torch.cuda.get_device_name(torch.cuda.current_device()))

# Leer modelo entrenado
model = YOLO(r"C:\Users\Donovan\Downloads\Git Scotch\modelos finales\best (1).pt")


# Lista de rutas de imágenes


imagen_paths = [
    r"C:\Users\Donovan\Downloads\redOne.png",
    r"C:\Users\Donovan\Downloads\cuadrados\Squares\50% White\50%\grid_15.png",
    r"C:\Users\Donovan\Downloads\cuadrados\Squares\30% White\70%\grid_16.png"
]


# Mapeo de etiquetas a porcentajes
percentage_map = {
    0: "10%",
    1: "20%",
    2: "30%",
    3: "40%",
    4: "50%",
    5: "60%",
    6: "70%",
    7: "80%",
    8: "90%"
}

# Procesar cada imagen en la lista
for imagen_path in imagen_paths:
    # Leer la imagen desde un archivo
    frame = cv2.imread(imagen_path)

    # Asegúrate de que la imagen se haya cargado correctamente
    if frame is None:
        print(f"Error al cargar la imagen: {imagen_path}")
        continue
    
    # Redimensionar la imagen
    scale_percent = 50  # Cambia este valor según tus necesidades
    new_width = int(frame.shape[1] * scale_percent / 100)
    new_height = int(frame.shape[0] * scale_percent / 100)
    new_size = (new_width, new_height)
    resized_frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
    
    # Realizar la predicción con el modelo
    resultados = model.predict(resized_frame, imgsz=640)
    
    # Obtener las anotaciones
    boxes = resultados[0].boxes  # Obtener las cajas de predicción
    
    # Dibujar las anotaciones manualmente
    for box in boxes:
        # Obtener coordenadas de la caja
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = int(box.cls[0])  # Obtener la etiqueta de clase
        
        # Obtener el porcentaje correspondiente a la etiqueta
        percentage = percentage_map.get(label, "Unknown")
        
        # Dibujar la caja
        cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Dibujar la etiqueta (solo con el porcentaje)
        cv2.putText(resized_frame, percentage, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostrar la imagen redimensionada con las anotaciones
    cv2.imshow(f"Clasificar pruebas scotch - {imagen_path}", resized_frame)

# Esperar hasta que se presione una tecla para cerrar todas las ventanas
cv2.waitKey(0)
cv2.destroyAllWindows()
