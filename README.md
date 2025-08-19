# Eye Gaze Estimation Using a Webcam  

## Overview  
This project focuses on the development of open-source software for real-time eye gaze estimation using a standard webcam. The primary goal is to create a **low-cost, accessible solution** for tracking eye movement during reading tasks without the need for specialized infrared hardware.  

Applications include:  
- Cognitive and educational research  
- Dyslexia and attention studies  
- User interface optimization  
- Accessibility enhancement  

Unlike commercial infrared systems, this project works with a **regular webcam**, making eye-tracking more affordable and portable.  

---

## Goals  
- Implement feature-based gaze estimation software using webcam input.  
- Enable accurate tracking of eye movements during text reading tasks.  
- Provide **interactive calibration and validation** for improved accuracy.  
- Compare **multiple mapping methods**:  
  - Linear mapping  
  - Polynomial regression  
  - Support Vector Regression (SVR)  
- Benchmark against **GazeRecorder**, a known tool in the domain.  
- Provide a modular framework for future research and parameter tuning.  

---

## Project Structure  

The workflow consists of three main stages:  

1. **Calibration** – The user fixates on predefined screen points to collect iris landmark data.  
2. **Validation** – Metrics (accuracy, precision, error in px/cm/deg) are computed to verify model performance.  
3. **Gaze Estimation** – The trained model runs in real-time and estimates gaze points as the user reads text.  

Each stage is implemented as a separate **class with threading support**, ensuring modularity and smooth performance.  

---

## Architecture & Configuration  

The system follows an **object-oriented, multithreaded architecture**:  

- `camera_feed.py` → Captures webcam frames (configurable via `cam_config.py`).  
- `calibration.py` → Guides the user through a calibration sequence.  
- `validation.py` → Computes metrics to evaluate model accuracy.  
- `gaze_part.py` → Runs real-time gaze estimation and visualization.  
- `constants.py` → Contains core parameters:  
cam_config.py → Holds resolution and FPS parameters for the webcam.

This modular design allows experimenting with different parameters and models by changing only configs, without touching the main code.

### Evaluation

The system was benchmarked across different frame rates and methods.
Metrics used: mean error, std deviation, p95 (95th percentile), stability.

### Findings:

Linear mapping → stable but less precise.
Polynomial regression → more accurate but sensitive to noise.
Lower FPS (30) showed better stability, while higher FPS (60) produced occasional large spikes in error (higher p95).

- p95 metric: 95% of errors are below this threshold, showing how often extreme deviations occur.

### Results & Visualizations

Calibration path plots
Scatter plots of gaze predictions
XY frame analysis
Metrics CSV files (metrics_summary.csv, metrics_stability.csv)
These can be reused to compare future improvements and tuning experiments.

### Usage

Configure camera parameters in cam_config.py.

Run the main script:
1. Configure camera parameters in cam_config.py.
2. Run the main script:
- `python main.py`
3. Follow on-screen calibration points.
4. Validate performance and start real-time gaze estimation.
