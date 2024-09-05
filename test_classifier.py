import cv2 as cv
import mediapipe as mp
import pickle
import numpy as np
import time
from collections import Counter
from sklearn.metrics import accuracy_score

# Load the model
model_dict = pickle.load(open('./final_model.p', 'rb'))
model = model_dict['model']

# Load the test data and labels
test_data_dict = pickle.load(open('./data.pickle', 'rb'))  # Assuming you have a test dataset
test_data = np.asarray(test_data_dict['data'])
test_labels = np.asarray(test_data_dict['labels'])

# Create a dictionary for labels
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
               12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
               23: 'X', 24: 'Y', 25: 'Z'}

# Initialize variables for accuracy calculation
true_labels = []
predicted_labels = []

# Initialize video capture
cap = cv.VideoCapture(0)

# Initialize mediapipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Initialize timer variables
start_time = time.time()
interval = 10  # 10-second interval
# Initialize an empty list to store predicted letters within the interval
predicted_letters_within_interval = []

# Main loop
while True:
    data_aux = []
    x_ = []
    y_ = []
    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])
        predicted_label = labels_dict[int(prediction[0])]

        # Append true and predicted labels for accuracy calculation
        true_labels.append(test_labels[i])  # Assuming you're iterating through the test dataset
        predicted_labels.append(predicted_label)

        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv.putText(frame, predicted_label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv.LINE_AA)

        # Check if the 10-second interval has passed and print the most recurring text
        if time.time() - start_time >= interval:
            # Print the most recurring text
            if predicted_labels:
                most_common_label = Counter(predicted_labels).most_common(1)[0][0]
                print('Most Common Extracted Text:', most_common_label)
                predicted_letters_within_interval.append(most_common_label)

            # Reset the timer and predicted labels
            start_time = time.time()
            predicted_labels = []

    cv.imshow('frame', frame)
    
    if cv.waitKey(25) & 0xFF == ord('q'):
        break
# Combine all predicted letters within the interval into a single string
extracted_text = ''.join(predicted_letters_within_interval)
print('The String Gestured is:', extracted_text)

cap.release()
cv.destroyAllWindows()
