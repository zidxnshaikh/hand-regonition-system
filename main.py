import cv2
import mediapipe as mp
import pyautogui
import math
import time


class HandDetector:
    def __init__(self, max_hands=2, detection_confidence=0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def find_hands(self, frame, draw=True):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_frame)
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return frame

    def find_position(self, frame, hand_no=0):
        landmark_list = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_no]
            h, w, _ = frame.shape
            for id, lm in enumerate(hand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_list.append([id, cx, cy])
        return landmark_list

    def fingers_up(self, landmark_list):
        if not landmark_list:
            return []
        tip_ids = [4, 8, 12, 16, 20]
        fingers = []
        if landmark_list[tip_ids[0]][1] > landmark_list[tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1, 5):
            if landmark_list[tip_ids[id]][2] < landmark_list[tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def get_hand_count(self):
        if self.results.multi_hand_landmarks:
            return len(self.results.multi_hand_landmarks)
        return 0

    def get_hand_label(self, hand_no=0):
        if self.results.multi_handedness:
            return self.results.multi_handedness[hand_no].classification[0].label
        return None

def get_gesture_name(fingers):
    gestures = {
        (0, 0, 0, 0, 0): "Fist",
        (1, 1, 1, 1, 1): "Open Palm",
        (0, 1, 1, 0, 0): "Peace",
        (1, 0, 0, 0, 0): "Thumbs Up",
        (0, 1, 0, 0, 0): "Pointing",
    }
    return gestures.get(tuple(fingers), "Unknown")


def get_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Hand Landmarks", cv2.WINDOW_NORMAL)
    detector = HandDetector()

    last_click_time = 0
    click_cooldown = 1

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)   # mirror view, handedness match ke liye zaroori
        frame = detector.find_hands(frame)
        hand_count = detector.get_hand_count()

        if hand_count > 1:
            cv2.putText(frame, "Please show only ONE hand!", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        elif hand_count == 1:
            label = detector.get_hand_label(0)

            if label != "Right":
                cv2.putText(frame, "Please use your RIGHT hand", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                landmark_list = detector.find_position(frame)
                if landmark_list:
                    fingers = detector.fingers_up(landmark_list)
                    gesture = get_gesture_name(fingers)
                    cv2.putText(frame, gesture, (50, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                1.5, (0, 255, 0), 3)

                    thumb = landmark_list[4][1:]
                    index = landmark_list[8][1:]
                    middle = landmark_list[12][1:]

                    current_time = time.time()
                    if current_time - last_click_time > click_cooldown:
                        if get_distance(thumb, index) < 40:
                            pyautogui.click()
                            last_click_time = current_time
                            print("Left Click")
                        elif get_distance(thumb, middle) < 40:
                            pyautogui.click(button='right')
                            last_click_time = current_time
                            print("Right Click")

        cv2.imshow("Hand Landmarks", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()