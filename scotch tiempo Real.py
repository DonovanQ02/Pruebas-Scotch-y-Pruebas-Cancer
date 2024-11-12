from ultralytics import YOLO
import cv2

#Leer modelo entrenado

model = YOLO(r"C:\Users\Donovan\Downloads\best (1).pt")

#realizar captura de video

cap = cv2.VideoCapture(0)

#bluce

while True:
    #leer nuestros fotogramas
    ret, frame = cap.read()
    
    #leemos los frames con el modelo
    resultados = model.predict(frame, imgsz = 320)
    
    #mostramos los resultados
    
    anotaciones = resultados[0].plot()
    
    #mostramos nuestros fotogramass
    
    cv2.imshow("detectar tennis", anotaciones)
    
    #cerrar nuestro programa
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()