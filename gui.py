import tkinter as tk
from PIL import Image, ImageTk
import cv2 as cv
import mediapipe as mp
import pickle
import numpy as np
import time
from collections import Counter

def main():
    # Load the model
    model_dict = pickle.load(open('./final_model.p', 'rb'))
    model = model_dict['model']

    # Load the test data and labels
    test_data_dict = pickle.load(open('./data.pickle', 'rb'))  # Assuming you have a test dataset
    test_labels = np.asarray(test_data_dict['labels'])

    # Create a dictionary for labels
    labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
                   12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
                   23: 'X', 24: 'Y', 25: 'Z', 26: 'Space'}

    # Initialize variables for accuracy calculation
    true_labels = []
    predicted_labels = []
    labellist = []

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

    # Create a GUI window
    root = tk.Tk()
    root.title("Hand Gesture Recognition")

    # Get the screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Set the size of the tkinter window to match the size of the Chrome window
    root.geometry(f"{screen_width}x{screen_height}")

    # Create a panel to display the video feed
    panel = tk.Label(root)
    panel.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    # Label to display the most common label and the predicted string
    label_text = tk.StringVar()
    label_text.set("Most Common Label: \nPredicted String: ")
    label_display = tk.Label(root, textvariable=label_text, font=("Helvetica", 16))
    label_display.pack()

    # Function to update the GUI with the latest frame
    def update_frame():
        nonlocal cap, model, hands, start_time, interval, predicted_letters_within_interval, true_labels, predicted_labels

        _, frame = cap.read()
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            if len(results.multi_hand_landmarks) > 1:
                print("Recognition Complete")
                button_restart = tk.Button(root, text="Restart", command=restart_program)
                button_restart.pack(side=tk.LEFT, padx=10, pady=10)
                button_exit = tk.Button(root, text="Exit", command=root.destroy)
                button_exit.pack(side=tk.RIGHT, padx=10, pady=10)
                return

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                data_aux = []
                x_ = []
                y_ = []
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
                    x_.append(x)
                    y_.append(y)

                x1 = int(min(x_) * frame.shape[1]) - 10
                y1 = int(min(y_) * frame.shape[0]) - 10

                x2 = int(max(x_) * frame.shape[1]) - 10
                y2 = int(max(y_) * frame.shape[0]) - 10

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
                        if most_common_label=='Space':
                            labellist.append(' ')
                        else:
                            labellist.append(most_common_label)
                        predicted_string = ''.join(labellist)
                        label_text.set(f"Most Common Label: {most_common_label}\nPredicted String: {predicted_string}")
                        predicted_letters_within_interval.clear()
                        predicted_labels.clear()

                    # Reset the timer and predicted labels
                    start_time = time.time()

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(image=frame)
        panel.configure(image=frame)
        panel.image = frame
        panel.after(10, update_frame)

    # Function to restart the program
    def restart_program():
        nonlocal cap, start_time, predicted_letters_within_interval, true_labels, predicted_labels

        cap.release()
        cv.destroyAllWindows()

        # Reset variables
        start_time = time.time()
        labellist.clear()
        predicted_letters_within_interval.clear()
        true_labels.clear()
        predicted_labels.clear()

        # Reinitialize video capture
        cap = cv.VideoCapture(0)

        # Restart the GUI
        root.quit()
        root.destroy()
        main()

    # Start updating the GUI with the latest frame
    update_frame()

    root.mainloop()

    cap.release()
    cv.destroyAllWindows()

# Call the main function to start the program
main()

