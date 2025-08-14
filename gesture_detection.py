import cv2
import mediapipe as mp
import numpy as np

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def detect_hands(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, c = frame.shape
                for connection in self.mp_hands.HAND_CONNECTIONS:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    
                    start_point = hand_landmarks.landmark[start_idx]
                    end_point = hand_landmarks.landmark[end_idx]
                    
                    start_x, start_y = int(start_point.x * w), int(start_point.y * h)
                    end_x, end_y = int(end_point.x * w), int(end_point.y * h)
                    
                    cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 4)
                
                for lm in hand_landmarks.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
        
        return frame
    


def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    
    print("手部關鍵點檢測程式啟動...")
    print("按 'q' 鍵退出")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法讀取攝影機畫面")
            break
        
        frame = cv2.flip(frame, 1)
        
        frame = detector.detect_hands(frame)
        
        cv2.imshow('Hand Landmarks Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
