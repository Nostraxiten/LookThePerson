# LookThePerson

<div align="center">

</pre>

**Framework Avanzado de Control Multi-Plataforma por Visión Computarizada y Gestos**

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Platform Support](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey.svg?style=for-the-badge&logo=linux)](https://www.linux.org/)
[![MediaPipe](https://img.shields.io/badge/Models-MediaPipe%20Tasks-teal.svg?style=for-the-badge)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/Graphics-OpenCV-orange.svg?style=for-the-badge&logo=opencv)](https://opencv.org/)

</div>

---

## 👁️ Descripción General

**LookThePerson** es un framework de visión por computadora que convierte tu cámara web en una interfaz de control en tiempo real. Utilizando hasta **5 modelos distintos de IA** (MediaPipe), el programa rastrea simultáneamente tu cuerpo, manos, rostro y objetos en el entorno.

Lo que comenzó como un pequeño script para Windows se ha transformado en un sistema **cross-platform (Windows y Linux)** altamente interactivo. Permite activar funciones, cambiar modelos sobre la marcha y usar gestos físicos para controlar aplicaciones nativas, todo desde una interfaz HUD futurista sobrepuesta en tu cámara.

---

##  Demostración de Interfaz

<p align="center">
  <img width="800" alt="LookThePerson Interface Showcase" src="https://github.com/user-attachments/assets/ae0331fe-ac3e-4056-ae89-b33bfecfc9d9" style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);" />
</p>

---

##  Arquitectura y Flujo

```mermaid
graph TD
    A[Cámara Web OpenCV] --> B{Launcher Multi-OS}
    B -->|Windows| C1[ctypes, calc.exe, DSHOW]
    B -->|Linux| C2[xdotool, subprocess, V4L2]
    
    C1 & C2 --> D[Motor de Modelos AI]
    
    D --> E1[Pose Landmarker]
    D --> E2[Hand Landmarker]
    D --> E3[Face Mesh 478-pt]
    D --> E4[Face Detection]
    D --> E5[Object Detection COCO]
    
    E1 & E2 & E3 --> F[Detector de Gestos]
    F -->|Clap, T-Pose, Squat| G1[Acciones Corporales]
    F -->|Dedos, Paz, Puño| G2[Acciones de Manos]
    F -->|Sonrisa, Guiño| G3[Expresiones Faciales]
    
    G1 & G2 & G3 --> H[Interacción con Sistema]
    H --> I[Calculadora Nativa / YouTube / Teclado Virtual]
    
    D --> J[KeyHandler en Tiempo Real]
    J --> K[HUD Rendering y Overlay]
```

---

##  Modelos de Inteligencia Artificial Integrados

La herramienta descarga automáticamente estos modelos en el primer arranque:

| Modelo MediaPipe | Propósito | Características Extra |
| :--- | :--- | :--- |
| **Pose Landmarker** | Esqueleto corporal completo | Segmentación de silueta con tintado dinámico |
| **Hand Landmarker** | Seguimiento de 21 puntos por mano | Conteo de dedos y reconocimiento de signos |
| **Face Mesh** | Malla facial 3D de 478 puntos | Detección de expresiones y rastreo de iris/mirada |
| **Face Detection** | Detección rápida de rostros | Bounding boxes y 6 puntos clave (ojos, nariz, boca) |
| **Object Detection** | Reconocimiento de entorno | 80 clases COCO con colores categorizados |

---

##  Controles en Tiempo Real (Teclado)

Puedes activar y desactivar funciones al instante **mientras la cámara está encendida**:

> [!TIP]
> **Modos de Configuración Rápida:** Usa los números del `1` al `4` para cambiar el perfil de los modelos activos de golpe (Ej: `1` = Todo activo, `4` = Solo Cara).

| Tecla | Acción / Toggle | Tecla | Acción / Toggle |
| :---: | :--- | :---: | :--- |
| `M` | Alternar máscara de **Segmentación** corporal | `S` | Tomar **Screenshot** (Captura PNG) |
| `F` | Activar/Desactivar **Face Mesh** overlay | `R` | Iniciar/Parar **Grabación de Video** |
| `O` | Activar/Desactivar **Detección de Objetos** | `C` | Cambiar color del esqueleto (Random) |
| `D` | Activar/Desactivar **Detección Rápida de Caras**| `X` | Bloquear/Desbloquear control de Calculadora |
| `G` | Mostrar/Ocultar **Cuadrícula de Referencia** | `+ / -` | Ajustar confianza de detección de objetos |
| `H` | Alternar el **Panel HUD de Ayuda** lateral | `1 - 4` | Cambiar Modos (Completo/Pose/Manos/Cara)|
| `T` | Mostrar/Ocultar texto de **Telemetría** inferior | `Q / Esc` | **Salir** del programa de forma segura |
| `N` | Activar/Desactivar **Modo Nocturno** (Inverso)| `B` | Ocultar/Mostrar **Bounding Boxes** |

---

##  Gestos Físicos Mapeados

El sistema incluye detección algorítmica de múltiples estados corporales que interactúan directamente con el Sistema Operativo:

| Gesto Detectado | Categoría | Acción Ejecutada en el OS |
| :--- | :--- | :--- |
| **Brazos extendidos (T-Pose)** | Cuerpo | Abre la calculadora nativa del sistema (`calc.exe` o `gnome-calculator`) |
| **Brazos cruzados al pecho** | Cuerpo | Cierra la calculadora activa |
| **Ambas manos levantadas** | Cuerpo | Abre una nueva pestaña de YouTube en el navegador |
| **Aplauso rápido** | Cuerpo | Cambia aleatoriamente el color del cuerpo |
| **Manos abiertas (5 dedos x2)**| Manos | Limpia la pantalla de la calculadora (Envía `Escape`) |
| **Conteo de dedos (1-4)** | Manos | Envía el número correspondiente a la calculadora |
| **Puño cerrado (0 dedos)** | Manos | Envía el símbolo suma (`+`) a la calculadora |

> [!NOTE]
> Nuevos gestos disponibles en el motor (Squat, Tocarse la cabeza, Guiños, Sonrisas) están listos para ser mapeados a nuevas funciones en `looktheperson.py`.

---

##  Requisitos e Instalación

### Prerrequisitos Sistema
* **Windows:** Windows 10/11.
* **Linux:** Cualquier distro con X11/Wayland. Requiere `xdotool` para control de ventanas. (`sudo apt install xdotool`)
* **Python:** 3.10+
* **Cámara web funcional.**

### Instalación

```bash
# Clonar el repo
git clone https://github.com/nostraxiten/LookThePerson.git
cd LookThePerson

# Crear y activar entorno virtual (Recomendado)
python -m venv .venv
# En Windows: .venv\Scripts\activate
# En Linux: source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

---

##  Modo de Ejecución

Ejecuta el nuevo launcher principal:

```bash
python looktheperson.py
```

### Argumentos Opcionales

| Comando | Descripción |
| :--- | :--- |
| `--windowed` | Lanza la aplicación en una ventana redimensionable (por defecto es pantalla completa). |
| `--camera N` | Usa un índice de cámara diferente si tienes múltiples conectadas (Ej: `--camera 1`). |
| `--no-calculator` | Inicia con el bloqueo de calculadora activado desde el principio. |
| `--fps N` | Fuerza una tasa de refresco específica de captura. |
| `--width N` / `--height N` | Fuerza una resolución de cámara específica. |

---

##  Estructura del Framework Expansivo

```text
LookThePerson/
├── looktheperson.py          # Launcher principal multiplataforma
├── platforms/                # Abstracción del sistema (Windows/Linux)
├── models/                   # Wrappers de IA (Pose, Hands, Face, Objects)
├── gestures/                 # Lógica matemática de detección de gestos
├── actions/                  # Controladoras (Teclas, Macros, Grabación)
├── ui/                       # Renderizado (HUD, Grid, Night Mode)
├── screenshots/              # Auto-generado al pulsar 'S'
└── recordings/               # Auto-generado al pulsar 'R'
```

---

> [!WARNING]
> **Privacidad Local:** El análisis de visión por computadora se ejecuta 100% de manera local y fuera de línea en tu CPU. No se transmite ningún frame a internet. Los modelos se descargan una sola vez desde los servidores de Google.
