import cv2
import mediapipe as mp
import time


def find_all_positions(img, hand_landmarks):
    positions = {}
    for landmark_id, landmark in enumerate(hand_landmarks.landmark):
        pixel_x, pixel_y = find_position(img, landmark)
        positions[landmark_id] = (pixel_x, pixel_y)
    return positions


def find_position(img, landmark):
    display_height, display_width, display_channel = img.shape
    return int(display_width * landmark.x), int(display_height * landmark.y)


def within_tolerance(tolerance, **args: int):
    if len(args) > 2:
        return True
    sorted_positions = sorted(args.values())
    return (sorted_positions[-1] - sorted_positions[0]) <= tolerance


class HandDetector:
    def __init__(self,
                 static_image_mode=False,
                 max_num_hands=2,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5
                 ):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Setup variables
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=self.static_image_mode,
                                         max_num_hands=self.max_num_hands,
                                         min_detection_confidence=self.min_detection_confidence,
                                         min_tracking_confidence=self.min_tracking_confidence,
                                         model_complexity=1)
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, should_draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if should_draw:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return img, results, results.multi_hand_landmarks


def test():
    cap = cv2.VideoCapture(0)
    previous_time = 0
    detector = HandDetector()

    while True:
        success, img = cap.read()
        img, results, landmarks = detector.find_hands(img)

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255))

        cv2.imshow('Image', img)
        cv2.waitKey(1)
