import cv2
import time
import numpy as np

def apply_filters(frame):
    frame_float = frame.astype(np.float32) / 255.0
    frame_float *= 0.9
    frame = (frame_float * 255).astype(np.uint8)

    blur = cv2.GaussianBlur(frame, (0, 0), 3)
    sharp = cv2.addWeighted(frame, 1.5, blur, -0.5, 0)

    hsv = cv2.cvtColor(sharp, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return enhanced

def main():
    cap = cv2.VideoCapture('/dev/video0')  # USB camera on Pi
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera resolution: {width}x{height}")
    
    cv2.namedWindow('USB Camera', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('USB Camera', 1280, 720)
    
    print("Camera opened successfully. Press 'q' to quit.")
    print("Press 'f' to toggle filters on/off")
    
    filters_enabled = True
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting...")
            break
        
        if filters_enabled:
            frame = apply_filters(frame)
        
        cv2.imshow('USB Camera', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            filters_enabled = not filters_enabled
            print("Filters:", "ON" if filters_enabled else "OFF")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
