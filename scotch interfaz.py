from ultralytics import YOLO
import cv2
import torch
import flet as ft
from flet import FilePickerResultEvent, FilePicker
import base64
import threading

# Verificar disponibilidad de CUDA
print("CUDA disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Procesador:", torch.cuda.get_device_name(torch.cuda.current_device()))

# Leer modelo entrenado
model = YOLO(r"C:\Users\Donovan\Downloads\best (1).pt")  # Ruta al modelo entrenado

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

# Generar una imagen base64 vacía
logo_path = "C:\\Users\\Donovan\\Downloads\\Logo.png"
logo_image = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)  # Leer con el canal alfa
_, buffer = cv2.imencode('.png', logo_image)
empty_image = base64.b64encode(buffer).decode()

def main(page: ft.Page):
    page.title = "Pruebas Scotch"
    page.window_width = 400
    page.window_height = 650
    image_display = ft.Image(src_base64=empty_image)  # Inicializar con una imagen vacía
    progress_bar = ft.ProgressBar(width=400, visible=False)  # Barra de progreso invisible por defecto
    scale_input = ft.TextField(label="Redimensionar imagen (%)", value="50")
    selected_image_path = None  # Variable para almacenar la ruta de la imagen seleccionada
    
    def on_file_picked(e: FilePickerResultEvent):
        nonlocal selected_image_path
        if e.files:
            selected_image_path = e.files[0].path
            preview_image(selected_image_path)
    
    def preview_image(image_path):
        # Leer la imagen desde un archivo
        frame = cv2.imread(image_path)
        
        # Asegúrate de que la imagen se haya cargado correctamente
        if frame is None:
            print(f"Error al cargar la imagen: {image_path}")
            return
        
        # Convertir la imagen a base64 para previsualización
        _, buffer = cv2.imencode('.png', frame)
        img_str = base64.b64encode(buffer).decode()
        
        # Mostrar la imagen en la interfaz de Flet
        image_display.src_base64 = img_str
        page.update()

    def process_image(image_path):
        # Leer la imagen desde un archivo
        frame = cv2.imread(image_path)

        # Asegúrate de que la imagen se haya cargado correctamente
        if frame is None:
            print(f"Error al cargar la imagen: {image_path}")
            return

        # Redimensionar la imagen
        scale_percent = int(scale_input.value)  # Obtener el valor desde el campo de texto
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

        # Convertir la imagen procesada a base64 para mostrarla en Flet
        _, buffer = cv2.imencode('.png', resized_frame)
        img_str = base64.b64encode(buffer).decode()

        # Mostrar la imagen en la interfaz de Flet
        image_display.src_base64 = img_str
        progress_bar.visible = False  # Ocultar la barra de progreso
        page.update()

        # Navegar a la pantalla de resultados
        page.go("/result")

    def process_image_async(image_path):
        progress_bar.visible = True  # Mostrar la barra de progreso
        page.update()
        threading.Thread(target=process_image, args=(image_path,)).start()

    file_picker = FilePicker(on_result=on_file_picked)

    def home_view():
        return ft.View(
            "/",
            controls=[
                ft.Column(
                    [
                        ft.Text("Seleccione la imagen a analizar"),
                        ft.ElevatedButton(
                            "Seleccionar Imagen",
                            on_click=lambda _: file_picker.pick_files(
                                allow_multiple=True
                            )
                        ),
                        file_picker,  # Asegurar que el file_picker esté en la vista
                        image_display,  # Previsualización de la imagen
                        ft.Text("Configuraciones:"),
                        scale_input,
                        ft.ElevatedButton(
                            "Procesar Imagen",
                            on_click=lambda _: process_image_async(selected_image_path)
                        ),
                        progress_bar  # Barra de progreso
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                )
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            vertical_alignment=ft.MainAxisAlignment.CENTER,
        )

    def result_view():
        return ft.View(
            "/result",
            controls=[
                ft.Column(
                    [
                        ft.Text("Imagen analizada:"),
                        image_display,
                        ft.ElevatedButton(
                            "Volver",
                            on_click=lambda _: page.go("/")
                        )
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                )
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            vertical_alignment=ft.MainAxisAlignment.CENTER,
        )

    def route_change(route):
        if page.route == "/":
            page.views.clear()
            page.views.append(home_view())
        elif page.route == "/result":
            page.views.clear()
            page.views.append(result_view())
        page.update()

    page.on_route_change = route_change
    page.go("/")

#ft.app(target=main)
ft.app(target=main, view=ft.WEB_BROWSER)

