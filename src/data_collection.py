import cv2
import numpy as np
import mediapipe as mp
import os

# Define gesture labels
sign_names = ["Hello", "Yes", "No", "Thank You", "Please", "Help", "Sorry",
              "I Love You (ILY)", "Goodbye", "Stop", "Start", "Finish",
              "Like", "Dislike", "Wait", "Go", "Come", "Eat", "Drink",
              "Sleep", "Where", "What", "Who", "How", "Why", "Because",
              "Want", "Need", "Feel", "Think", "Time", "Again",
              "Different", "More", "Less"]

data = []
label_data = []

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5)

os.makedirs("data", exist_ok=True)

cap = cv2.VideoCapture(0)

for label_idx, label in enumerate(sign_names):
    print(f"Collecting data for: {label} (Press 'c' to capture, 'n' for next)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [lm.x for lm in hand_landmarks.landmark] + \
                            [lm.y for lm in hand_landmarks.landmark] + \
                            [lm.z for lm in hand_landmarks.landmark]

                cv2.putText(frame, f"Press 'c' to capture {label}",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2)

                if len(landmarks) == 63 and cv2.waitKey(1) & 0xFF == ord('c'):
                    data.append(landmarks)
                    label_data.append(label_idx)
                    print(f"Captured {label}")

        cv2.imshow("Sign Data Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord('n'):
            break

cap.release()
cv2.destroyAllWindows()

np.save("data/sign_language_data.npy", np.array(data))
np.save("data/sign_language_labels.npy", np.array(label_data))

print("Dataset saved successfully!")
