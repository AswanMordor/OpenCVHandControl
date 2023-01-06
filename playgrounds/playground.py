import cv2
import mediapipe as mp
import time

# Setup variables
cap = cv2.VideoCapture(0)
mphands = mp.solutions.hands
hands = mphands.Hands()
mpdraw = mp.solutions.drawing_utils

current_time = 0
previous_time = 0
tolerance = 40

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            thumb_x, thumb_y = 0, 0
            index_x, index_y = 0, 0
            for id, landmark in enumerate(hand_landmarks.landmark):
                # print(id, landmark)
                display_height, display_width, display_channel = img.shape
                pixel_x, pixel_y = int(display_width * landmark.x), int(display_height * landmark.y)
                # print('Landmark with ID: ', str(id), ' at x: ', str(pixel_x), ' y: ' + str(pixel_y))
                if id == 4:
                    thumb_x, thumb_y = pixel_x, pixel_y
                if id == 8:
                    index_x, index_y = pixel_x, pixel_y
            if index_x in range(thumb_x-tolerance, thumb_x+tolerance) and index_y in range(thumb_y-tolerance, thumb_y+tolerance):
                cv2.putText(img, 'BOOM', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255))
            mpdraw.draw_landmarks(img, hand_landmarks, mphands.HAND_CONNECTIONS)

    current_time = time.time()
    fps = 1 / (current_time-previous_time)
    previous_time = current_time
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255))

    cv2.imshow('Image', img)
    cv2.waitKey(1)
