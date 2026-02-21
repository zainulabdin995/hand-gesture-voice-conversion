import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading

# Load the pre-trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Gesture labels
labels_dict = {0: 'Hello', 1: 'Love you',2: 'Victory',3: 'Rock on', 4: 'strength', 
               5: 'Call me',6: 'Thumbs up', 7: 'Thumbs down', 8: 'Pointing up',9: 'Pointing down', 
               10: 'Good luck', 11: 'OK', 12: 'You', 13: 'Protest', 14: 'Pinch', 
               15: 'Hope', 16: 'Small amount', 17: 'Greetings',18: 'Question', 19: 'Shoot'}

# Initialize pyttsx3
engine = pyttsx3.init()

# Voice function
def speak_out(text):
    engine.say(text)
    engine.runAndWait()

def process_hand_gesture(frame, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            data_aux = []
            x_ = []
            y_ = []

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            if data_aux:
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10

                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

                # Speak out the recognized gesture using computer voice
                threading.Thread(target=speak_out, args=(predicted_character,)).start()

    return frame

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    processed_frame = process_hand_gesture(frame.copy(), results)

    cv2.imshow('frame', processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
