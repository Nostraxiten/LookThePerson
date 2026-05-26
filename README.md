# LookThePerson

Real-time body and hand gesture detection for Windows using MediaPipe Tasks and OpenCV.

This project reads webcam input to render full-body pose landmarks, hand skeletons, and gesture-based controls. It automatically downloads MediaPipe model assets on first run and uses native Windows APIs to interact with the calculator and browser.

## Features

- Full-body pose detection with skeleton overlay and segmentation tinting
- Hand landmark tracking with hand skeleton rendering
- Automatic download of MediaPipe `pose_landmarker_full.task` and `hand_landmarker.task`
- Gesture-driven Windows Calculator control
- Clap gesture changes the body tint color
- Live position tracking with coordinate output
- Fullscreen display with optional windowed mode

## Requirements

- Windows 10 or 11
- Python 3.10+
- Webcam
- `opencv-python`
- `mediapipe`

## Installation

1. Create a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

2. Install the dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

## Usage

Run the application:

```powershell
python hand.py
```

Optional command-line arguments:

- `--camera`: camera index (default: `0`)
- `--width`: requested frame width (default: `1280`)
- `--height`: requested frame height (default: `720`)
- `--fps`: requested camera FPS (default: `30`)
- `--windowed`: open the display in a standard window instead of fullscreen
- `--no-calculator`: disable gesture controls for the calculator

## Gesture Controls

- `Q` or `Esc`: exit the application
- `X`: lock/unlock calculator gesture controls
- Clap: change body tint color
- Arms open: open Windows Calculator
- Arms closed: close Windows Calculator
- Both hands open: clear calculator input
- Both hands raised: open YouTube in the browser

## Notes

- This script is designed specifically for Windows.
- If the model files are missing, they are downloaded automatically to the project folder.
- The `.task` model files are large and should not be committed to source control if you want to keep the repository lightweight.

## Files

- `hand.py`: main application script
- `hand_landmarker.task`: MediaPipe hand model asset (downloaded automatically)
- `pose_landmarker_full.task`: MediaPipe pose model asset (downloaded automatically)
- `requirements.txt`: Python dependencies
