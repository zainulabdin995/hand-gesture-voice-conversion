# Hand Gesture Recognition & Voice Conversion

## Overview
Designed to assist individuals with speech and hearing impairments, this project acts as a digital translator between visual sign language and audible speech. Utilizing webcams for image extraction, the system applies background removal and binary/grayscale conversion before feeding the frames into a Convolutional Neural Network (CNN) and Support Vector Machine (SVM) pipeline for classification. The predicted text is then dynamically converted into audio.



## Tech Stack
* **Language:** Python
* **Computer Vision:** OpenCV (Background removal, contour detection)
* **Machine Learning:** CNNs, SVM
* **Voice Synthesis:** Text-to-Speech (TTS) libraries

## How to Run
1. Clone the repository.
2. Install requirements: `pip install -r requirements.txt`
3. Execute the main vision script: `python src/gesture_recognizer.py`
4. Perform the trained gestures in front of your webcam to hear the audio output.
