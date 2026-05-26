LookThePerson
Detección en tiempo real de gestos corporales y de manos para Windows usando MediaPipe Tasks y OpenCV.
Este proyecto lee la entrada de la webcam para renderizar landmarks de pose corporal completa, esqueletos de manos y controles basados en gestos. Descarga automáticamente los modelos de MediaPipe en el primer uso y emplea APIs nativas de Windows para interactuar con la calculadora y el navegador.
Características

Detección de pose corporal completa con superposición de esqueleto y tintado por segmentación
Seguimiento de landmarks de manos con renderizado de esqueleto
Descarga automática de los modelos pose_landmarker_full.task y hand_landmarker.task de MediaPipe
Control de la Calculadora de Windows mediante gestos
El gesto de aplauso cambia el color del tintado corporal
Seguimiento de posición en tiempo real con salida de coordenadas
Pantalla completa con modo ventana opcional

<img width="662" height="386" alt="Captura de pantalla 2026-05-26 130327" src="https://github.com/user-attachments/assets/ae0331fe-ac3e-4056-ae89-b33bfecfc9d9" />


Requisitos

Windows 10 u 11
Python 3.10+
Webcam
opencv-python
mediapipe

Instalación

Crea un entorno virtual:

powershell   python -m venv .venv
   .\.venv\Scripts\activate

Instala las dependencias:

powershell   pip install -r requirements.txt
Uso
Ejecuta la aplicación:
powershellpython hand.py
Argumentos opcionales de línea de comandos:

--camera: índice de cámara (por defecto: 0)
--width: ancho de fotograma solicitado (por defecto: 1280)
--height: alto de fotograma solicitado (por defecto: 720)
--fps: FPS de cámara solicitados (por defecto: 30)
--windowed: abre la visualización en ventana estándar en lugar de pantalla completa
--no-calculator: desactiva los controles por gestos para la calculadora

Controles por Gestos

Q o Esc: salir de la aplicación
X: bloquear/desbloquear controles por gestos de la calculadora
Aplauso: cambiar el color del tintado corporal
Brazos abiertos: abrir la Calculadora de Windows
Brazos cerrados: cerrar la Calculadora de Windows
Ambas manos abiertas: limpiar la entrada de la calculadora
Ambas manos levantadas: abrir YouTube en el navegador

Notas

Este script está diseñado específicamente para Windows.
Si los archivos de modelos no están presentes, se descargan automáticamente en la carpeta del proyecto.
Los archivos de modelo .task son de gran tamaño y no deberían incluirse en el control de versiones si se quiere mantener el repositorio ligero.

Archivos

hand.py: script principal de la aplicación
hand_landmarker.task: modelo de manos de MediaPipe (descarga automática)
pose_landmarker_full.task: modelo de pose de MediaPipe (descarga automática)
requirements.txt: dependencias de Python
