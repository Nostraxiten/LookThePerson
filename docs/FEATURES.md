# LookThePerson — Feature Catalogue

Every capability in the project, grouped by subsystem. Modes are listed with
the key you use to reach them (`--mode <key>`, or `TAB` in the app).

---

## Modes (49)

### Detection (7)

| # | Mode | Key | What it does |
|---|------|-----|--------------|
| 1 | Complete | `full` | All five models running simultaneously |
| 2 | Pose only | `pose` | Skeleton with live joint angles drawn at each joint |
| 3 | Hands only | `hands` | Per-hand finger count, handedness and gesture name |
| 4 | Face only | `face` | Face mesh, expressions, gaze and head pose |
| 5 | Objects | `objects` | COCO detection with a running inventory of what was seen |
| 6 | Minimal | `minimal` | Clean video, no overlays — for recording |
| 7 | Crowd | `crowd` | Multi-person counting, per-person colour, peak occupancy |

### Fitness (6)

| # | Mode | Key | What it does |
|---|------|-----|--------------|
| 8 | Rep counter | `reps` | Counts reps of one selected exercise with a depth ring |
| 9 | Workout | `workout` | Detects several exercises at once, sets, rest, calories |
| 10 | Posture coach | `posture` | Live posture grade with specific corrective advice |
| 11 | Balance | `balance` | Centre of mass, stability score, hold timer |
| 12 | Stretching | `stretch` | Times held positions, reports left/right symmetry |
| 13 | Cardio | `cardio` | Movement intensity, effort zones, calorie estimate |

### Wellness (5)

| # | Mode | Key | What it does |
|---|------|-----|--------------|
| 14 | Drowsiness | `drowsiness` | PERCLOS fatigue monitoring, blinks, yawns, alarm |
| 15 | Focus | `focus` | Attention-gated Pomodoro; only focused time counts |
| 16 | Breathing | `breathing` | Animated 4-7-8 pacer plus measured breathing rate |
| 17 | Ergonomics | `ergonomics` | Posture, screen distance and 20-20-20 break reminders |
| 18 | Meditation | `meditation` | Timer that advances only while you stay still |

### Creative (15)

| # | Mode | Key | What it does |
|---|------|-----|--------------|
| 19 | Night vision | `night_vision` | Green intensifier with a sweeping scan line |
| 20 | Thermal | `thermal` | False-colour heat map |
| 21 | Matrix | `matrix` | Falling code with your silhouette cut through it |
| 22 | ASCII | `ascii` | Whole scene rendered as coloured characters |
| 23 | Cartoon | `cartoon` | Flat colour regions with ink outlines |
| 24 | Sketch | `sketch` | Pencil drawing |
| 25 | Glitch | `glitch` | Digital corruption that scales with how much you move |
| 26 | Silhouette | `silhouette` | Flat figure on a flat background, 5 palettes |
| 27 | Hologram | `hologram` | Tinted, scanlined sci-fi projection |
| 28 | Spotlight | `spotlight` | Follow-spot lighting with an edge halo |
| 29 | Ghost | `ghost` | Motion echo of past silhouettes |
| 30 | X-ray | `xray` | High-contrast radiograph look with skeleton |
| 31 | Trails | `trails` | Light trails on hands, feet and head |
| 32 | Heat map | `heatmap` | Accumulates where movement happens over time |
| 33 | Particles | `particles` | Particles emitted from the hands, reactive to speed |

### Interaction (6)

| # | Mode | Key | What it does |
|---|------|-----|--------------|
| 34 | Air drawing | `airdraw` | Draw with your fingertip; pinch to lift the pen |
| 35 | Air mouse | `airmouse` | Cursor control by finger, pinch to click (permissioned) |
| 36 | Virtual piano | `piano` | On-screen keys played with fingertips |
| 37 | Media control | `media` | Play/pause and track skipping by gesture (permissioned) |
| 38 | Presentation | `presentation` | Slide navigation by pointing, plus a laser dot |
| 39 | Hand numbers | `signs` | Reads numbers from fingers, pinch to confirm |

### Utility (10)

| # | Mode | Key | What it does |
|---|------|-----|--------------|
| 40 | Privacy | `privacy` | Face blur/pixelate or full-body masking, 4 levels |
| 41 | Security | `security` | Surveillance station: night vision, subject identification, zone alarms ([details](#security-mode)) |
| 42 | Presence | `presence` | Attendance tracking: present/absent time, absences |
| 43 | Virtual background | `greenscreen` | Chroma, blur, gradient or your own image |
| 44 | Photo booth | `photobooth` | Gesture-triggered countdown and burst |
| 45 | Timelapse | `timelapse` | Periodic capture at an adjustable interval |
| 46 | Measurements | `measure` | Body proportions in centimetres from a height calibration |
| 47 | Calibration | `calibration` | Framing guides, visibility and lighting diagnostics |
| 48 | Debug | `debug` | Landmark indices, stage timings, event counters |
| 49 | Benchmark | `benchmark` | Measures the FPS cost of each model combination |

---

## Security mode

`security` is the one mode that takes over the whole picture. Every decorative
overlay is switched off and the shared HUD steps aside, because a surveillance
frame is evidence and anything painted over it that is not information is in
the way.

```bash
python looktheperson.py --mode security
```

### Night vision

Metered sensor emulation, switching the way a real installation does.

| Mode | What it does |
|------|--------------|
| `auto` | Meters the scene and moves between `dia` and `ir` on a hysteresis band, so a camera on a doorway does not oscillate at dusk |
| `dia` | Daylight: colour, gentle exposure lift only |
| `ir` | Monochrome infrared: local-contrast boost, IR-illuminator radial falloff, sensor grain |
| `intensificador` | Image-intensifier tube: green phosphor, heavy gain, highlight bloom, scan structure |
| `termico` | False-colour heat palette — useful for separating a body from a dark background |
| `realce` | Lifts shadows while keeping colour, for when you care what colour the jacket was |

Auto-gain (AGC) targets a mid-grey average and is clamped, so a nearly black
frame cannot amplify pure noise to full scale. Gain can also be driven by hand.

> These modes make an underexposed frame legible. None of them recover detail
> the sensor never captured, and `termico` maps brightness, not temperature.

### Subject identification

Each person in view gets a persistent `SUJ-nn` label, a build code, a dwell
timer, a confidence score and a stature estimate. Identity is carried by two
signals: box overlap and centroid distance link a detection to the track it
continues, and a **body signature** — limb lengths normalised by torso, so the
vector barely moves as someone walks toward the camera — recovers the identity
after a subject leaves the frame and returns.

> Scope: this tells apart the few people currently in view, from skeleton
> proportions. It carries no identity between sessions, stores no face data,
> and cannot recognise a stranger. The stature figure is a proportional
> estimate, not a measurement.

### Zones, events and the operator display

**Zones** — `PERIMETRO`, `CENTRO`, `IZQUIERDA`, `DERECHA`, `UMBRAL` (a doorway
strip). An identified subject entering the active zone while armed raises an
intrusion.

**Events** — `ALTA` (new subject), `REGRESO` (re-identified), `BAJA` (left),
`INTRUSION`, `MULTIPLE`. Shown on screen newest-last and exportable to JSON.
Presence is debounced, so a one-frame false detection cannot trip the alarm.

**DVR overlay** — channel label, burnt-in timestamp, arm state with a 10 s
arming delay, recording indicator, sensor readout (mode, lux, gain), subject
counter, signal bars, and per-subject targeting reticles with identity cards.
Three OSD detail levels.

### Keys

| Key | Action | Key | Action |
|:---:|--------|:---:|--------|
| `A` | Arm / disarm | `I` | Subject identification overlay |
| `N` | Night-vision mode | `L` | Event log panel |
| `J` | Auto-gain | `O` | OSD detail level |
| `+` `-` | Gain up / down | `E` | Export the log to JSON |
| `X` | Detection zone | `Z` | Clear the log |
| `C` | Camera channel | `0` | Reset the whole system |

---

## Analytics (30)

**Joint angles** — 50. 12 named joints · 51. symmetric averages · 52. trunk
inclination · 53. shoulder tilt · 54. hip tilt · 55. head tilt · 56. body
orientation (front/left/right) · 57. limb lengths · 58. implausible-angle
flagging.

**Posture** — 59. 0-100 score with letter grade · 60. slouch detection ·
61. forward-head detection · 62. uneven-shoulder detection · 63. trunk-lean
detection · 64. per-issue severity and advice · 65. debounced alerts ·
66. session statistics.

**Exercise** — 67. hysteresis rep state machine · 68. 12-exercise catalogue ·
69. rep depth measurement · 70. form scoring with per-exercise checks ·
71. tempo measurement · 72. automatic set closing on rest · 73. simultaneous
multi-exercise counting · 74. MET calorie estimation · 75. intensity zones with
time-in-zone.

**Motion and balance** — 76. per-landmark velocity · 77. whole-body motion
energy · 78. travel direction · 79. centre of mass · 80. postural sway ·
81. stability score · 82. left/right symmetry · 83. dominant side · 84. landmark
trajectories · 85. stillness detection · 86. jump-height estimate.

**Face** — 87. eye aspect ratio · 88. blink detection and rate · 89. PERCLOS ·
90. yawn detection · 91. drowsiness levels · 92. head pose (yaw/pitch/roll) ·
93. gaze direction · 94. attention tracking · 95. focus streaks.

---

## Gestures (34)

**Body (17)** — clap, arms open, arms closed, both hands raised, head touch,
T-pose, squat, one hand raised (L/R), leaning (L/R), arms crossed, hands on
hips, pointing (L/R), jumping, sitting, lying down, kicking (L/R), hand to face.

**Hands (14)** — fist, open palm, thumbs up, thumbs down, peace, rock, OK,
pointing, call me, gun, Spock, pinch, three, four — plus continuous hand
openness, orientation and finger counting.

**Face (7)** — mouth open, left/right eye closed, left/right wink, eyebrows
raised, smile.

All gestures pass through debouncing and cooldowns, and any of them can be
bound to any action in configuration.

---

## Visual effects (40)

**Filters (23)** — none, invert, grayscale, night vision, thermal, sepia,
duotone, posterize, edges, sketch, cartoon, emboss, sharpen, blur, pixelate,
vignette, scanlines, chromatic aberration, glitch, bloom, ASCII, colour pop,
kaleidoscope. Cached lookup tables and masks keep the expensive ones real-time.

**Background (11)** — mask preparation with feathering, background blur,
image replacement, flat colour/chroma, silhouette, cutout, spotlight with halo,
person outline, hologram, ghost trail, region blur/pixelate for privacy.

**Overlays (6)** — 7 skeleton styles (classic, glow, dots, thick, wire, neon,
bones), motion trails, motion heat map with hotspot and coverage, particle
system, radar scope, progress rings, scan lines, landmark labels.

---

## Interface (15)

Mode badge and status line · colour-coded FPS · active-model list · blinking
recording indicator · detection counts · gesture readout · mode-supplied HUD
lines · paginated help panel with live ON/OFF state · mode picker overlay ·
toast notifications with severity and fade · FPS sparkline with target
reference · profiler breakdown bars · 7 colour themes · grid, thirds, safe-area
and crosshair guides · corner-bracket detection boxes.

---

## Capture and I/O (14)

Screenshots with collision-free naming · photo bursts · video recording with
codec fallback · animated GIF export (written from scratch, no extra
dependency) · contact-sheet montages · capture history and counts · free-space
reporting · camera enumeration · resolution probing · live camera switching ·
video-file and image-folder input · automatic reconnection on device loss ·
mirror toggle · session export to JSON, CSV and JSONL.

---

## Core infrastructure (20)

JSON configuration with user, project and profile layers · CLI overrides ·
live save · dotted-path get/set · event bus with wildcards, history and handler
isolation · One-Euro, exponential, median and point filters · hysteresis ·
debouncing · edge detection · cooldowns · velocity tracking · ring buffers with
percentiles · FPS tracking with 1% lows and jitter · per-stage profiler ·
performance monitor with degradation suggestions · landmark smoothing ·
model striding · lazy model loading · headless operation.

---

## Platform integration (12)

Windows and Linux bridges · monitor geometry · camera backend selection ·
calculator launch/close · window find/move/close by title or PID · mouse
movement, clicking and scrolling · media keys · desktop notifications ·
URL opening · graceful capability degradation when a tool is missing.

---

## Quality

- **271 unit tests**, covering geometry, filters, config, events, angles,
  posture, rep counting, fitness, motion, face metrics, session export, the
  gesture engine, the action layer and all 49 modes (each driven through both
  a populated frame and an empty one).
- **Three runtime dependencies**: OpenCV, MediaPipe, numpy.
- **Runs offline**: models download once, then everything is local.
