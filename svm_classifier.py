import os
import pickle
import mediapipe as mp
import cv2 as cv
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load hand landmark data
data_dir = './data'
data = []
labels = []

for dir_ in os.listdir(data_dir):
    for img_path in os.listdir(os.path.join(data_dir, dir_)):
        data_aux = []

        img = cv.imread(os.path.join(data_dir, dir_, img_path))
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.4)
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)

        # Fill in missing values with placeholders (e.g., -1)
        max_landmarks = 42  # Assuming 21 landmarks with x, y coordinates
        while len(data_aux) < max_landmarks * 2:
            data_aux.extend([-1, -1])

        data.append(data_aux)
        labels.append(dir_)

# Convert data and labels to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize and train the SVM classifier
svm_model = SVC(kernel='linear')  # Linear kernel
svm_model.fit(x_train, y_train)

# Evaluate the model
y_predict = svm_model.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)
print("Accuracy:", accuracy)

# Save the trained model
with open('svm_model.p', 'wb') as f:
    pickle.dump({'model': svm_model}, f)
