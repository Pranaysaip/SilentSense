import os
import pickle
import mediapipe as mp
import cv2 as cv
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

data_dir = './data'

data = []
labels = []

for dir_ in os.listdir(data_dir):
    for img_path in os.listdir(os.path.join(data_dir, dir_)):
        data_aux = np.zeros((21, 2))  # Initialize data_aux as a numpy array
        
        img = cv.imread(os.path.join(data_dir, dir_, img_path))
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img_rgb,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Extract x and y coordinates of hand landmarks
                landmarks_xy = np.array([[landmark.x, landmark.y] for landmark in hand_landmarks.landmark])

                # Fill data_aux with landmark coordinates
                data_aux[:len(landmarks_xy)] = landmarks_xy

        data.append(data_aux.flatten())  # Flatten data_aux before appending to data
        labels.append(dir_)

# Save data and labels
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

# Load and print the length of the loaded data
data_dict = pickle.load(open('./data.pickle', 'rb'))
print(len(data_dict))
