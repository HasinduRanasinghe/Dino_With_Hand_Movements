import cv2
import mediapipe as mp
import pyautogui
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("Switch to Chrome and open the Dino game (chrome://dino)")
time.sleep(5)

pyautogui.press('space')


def is_hand_folded(landmarks):
    folded = True
    finger_tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP,
                   mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                   mp_hands.HandLandmark.RING_FINGER_TIP,
                   mp_hands.HandLandmark.PINKY_TIP]
    finger_base_knuckles = [mp_hands.HandLandmark.INDEX_FINGER_PIP,
                            mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                            mp_hands.HandLandmark.RING_FINGER_PIP,
                            mp_hands.HandLandmark.PINKY_PIP]

    for tip, base in zip(finger_tips, finger_base_knuckles):
        if landmarks[tip].y < landmarks[base].y:
            folded = False

    return folded


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if is_hand_folded(hand_landmarks.landmark):
                pyautogui.press('space')

    cv2.imshow('Hand Motion Controller', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()