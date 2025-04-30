### Eye Gaze Estimation Using a Webcam
## Overview
This project focuses on the development of open-source software for real-time eye gaze estimation using a standard webcam. The primary goal is to create a low-cost, accessible solution for tracking eye movement during reading tasks without the need for expensive, specialized hardware. The software aims to support cognitive and educational research, user interface optimization, and accessibility studies by enabling the detection of the user's gaze points as they interact with textual content.

Eye tracking is a powerful technique for understanding cognitive processes, visual attention, and user engagement. Traditional systems often rely on infrared-based equipment and are limited to laboratory settings due to their cost and complexity. In contrast, this project uses only a standard optical webcam, making eye-tracking research and applications more affordable and portable.

## Goals
- Develop a feature-based gaze estimation software using webcam input.

- Enable accurate tracking of eye movement during text reading tasks.

- Provide an interactive calibration and validation process for improved accuracy.

- Compare at least two gaze estimation methods.

- Benchmark the developed software against GazeRecorder, a well-known tool in the domain.

- Make the tool accessible for future research, especially in areas like dyslexia studies, user interface design, and attention analysis.

## Project Structure
The software workflow consists of three main stages:

- Calibration – The user is guided to fixate on predefined screen points to collect reference data.

- Validation – The collected data is used to verify and adjust the gaze estimation model.

- Gaze Estimation – The trained model estimates the user's gaze position in real-time as they read text.

The software includes features to assist users during calibration, ensuring better data quality and usability, even outside of controlled lab environments.
