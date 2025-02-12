import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Start webcam
cap = cv2.VideoCapture(0)  # Use webcam

while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Run YOLO detection
        results = model(frame)
        
        # Get detections
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])  # Get class ID
                confidence = box.conf[0]    # Get confidence score
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box

                # Only draw bounding box if the detected object is a person (class ID 0)
                if class_id == 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
                    cv2.putText(frame, f"Person {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the annotated frame
        cv2.imshow("Human Detection", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
