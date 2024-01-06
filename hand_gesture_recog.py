import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

# Load model
with open('model.pkl', 'rb') as f:
    svm = pickle.load(f)

def image_processed(hand_img):
    # Image processing
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
    img_flip = cv2.flip(img_rgb, 1)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)
    output = hands.process(img_flip)
    hands.close()

    try:
        data = output.multi_hand_landmarks[0]
        data = str(data)
        data = data.strip().split('\n')

        garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']
        without_garbage = [i for i in data if i not in garbage]

        clean = [i.strip()[2:] for i in without_garbage]

        for i in range(0, len(clean)):
            clean[i] = float(clean[i])
        return clean
    except:
        return np.zeros([1, 63], dtype=int)[0]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

last_prediction_time = time.time()
subtitle_duration = 5  # seconds

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    data = image_processed(frame)
    data = np.array(data)
    y_pred = svm.predict(data.reshape(-1, 63))

    # Display predicted letter in the lower part of the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, frame.shape[0] - 50)  # Adjust Y coordinate for lower part
    fontScale = 1  # Smaller font size
    color = (255, 255, 255)  # White text
    thickness = 2

    # Create black background
    black_bg = np.zeros_like(frame)
    black_bg = cv2.putText(black_bg, str(y_pred[0]), org, font, fontScale, color, thickness, cv2.LINE_AA)

    # Combine the original frame and the black background with text
    frame = cv2.addWeighted(frame, 1, black_bg, 0.5, 0)

    # Check if hand is detected
    if np.all(data == 0):
        # If no hand detected, check if the subtitle duration has passed
        if time.time() - last_prediction_time >= subtitle_duration:
            # Clear subtitle if duration has passed
            black_bg = np.zeros_like(frame)
            frame = cv2.addWeighted(frame, 1, black_bg, 0.5, 0)
    else:
        # Update the time of the last prediction
        last_prediction_time = time.time()

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
