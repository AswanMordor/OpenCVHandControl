import math
import time
import cv2
import hand_tracking_module as hand_tracking
import osascript
from pynput.keyboard import Key, Controller

# Fingers
THUMB_TIP = 4
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_TIP = 12
RING_FINGER_TIP = 16
PINKY_TIP = 20

# Colors
PURPLE = (255, 0, 255)
WHITE = (255, 255, 255)
BLUE = (162, 118, 19)
LIGHT_GREEN = (162, 222, 19)

# Camera
CAMERA_WIDTH, CAMERA_HEIGHT = 1280, 720

# Tolerances
MUTE_TOLERANCE = 80
SIGN_TOLERANCE = 100

cap = cv2.VideoCapture(0)
cap.set(CAMERA_WIDTH, CAMERA_HEIGHT)
previous_time = 0
detector = hand_tracking.HandDetector(min_detection_confidence=0.7)
keyboard = Controller()

paused = False

while True:
    success, img = cap.read()
    img, results, hands = detector.find_hands(img)

    if hands:
        for landmarks in hands:
            positions = hand_tracking.find_all_positions(img, landmarks)

            thumb_tip_x, thumb_tip_y = positions[THUMB_TIP]
            index_tip_x, index_tip_y = positions[INDEX_FINGER_TIP]
            middle_tip_x, middle_tip_y = positions[MIDDLE_FINGER_TIP]
            ring_tip_x, ring_tip_y = positions[RING_FINGER_TIP]
            pinky_tip_x, pinky_tip_y = positions[PINKY_TIP]
            center_x, center_y = (thumb_tip_x + index_tip_x) // 2, (thumb_tip_y + index_tip_y) // 2

            length = math.hypot(thumb_tip_x - index_tip_x, thumb_tip_y - index_tip_y)

            if (
                hand_tracking.within_tolerance(SIGN_TOLERANCE, ring_tip_x, thumb_tip_x, middle_tip_x) and
                hand_tracking.within_tolerance(SIGN_TOLERANCE, ring_tip_y, thumb_tip_y, middle_tip_y)
            ):
                center_circle_color = LIGHT_GREEN if length < MUTE_TOLERANCE else BLUE
                cv2.circle(img, (center_x, center_y), 10, center_circle_color, cv2.FILLED)
                cv2.circle(img, (center_x, center_y), 15, center_circle_color, cv2.BORDER_DEFAULT)
                cv2.circle(img, (thumb_tip_x, thumb_tip_y), 15, PURPLE, cv2.FILLED)
                cv2.circle(img, (index_tip_x, index_tip_y), 15, PURPLE, cv2.FILLED)
                cv2.line(img, positions[THUMB_TIP], positions[INDEX_FINGER_TIP], PURPLE, 3)
                cv2.putText(img, str(length)[:4],
                            (center_x + 20, center_y + 10),
                            cv2.FONT_HERSHEY_PLAIN, 3,
                            BLUE, thickness=5)
                osascript.osascript("set volume output volume {}".format(length / 3))  # this only works for Mac

                if (
                        math.fabs(pinky_tip_x - thumb_tip_x) <= SIGN_TOLERANCE + 30 and
                        math.fabs(pinky_tip_y - thumb_tip_y) <= SIGN_TOLERANCE
                ):
                    if not paused:
                        cv2.circle(img, (pinky_tip_x, pinky_tip_y), 15, PURPLE, cv2.FILLED)
                        keyboard.press(Key.space)
                        keyboard.release(Key.space)
                        paused = True
                elif paused:
                    cv2.circle(img, (pinky_tip_x, pinky_tip_y), 15, WHITE, cv2.FILLED)
                    keyboard.press(Key.space)
                    keyboard.release(Key.space)
                    paused = False

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, WHITE)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
