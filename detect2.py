import cv2
from ultralytics import YOLO
from collections import Counter
from picamera2 import Picamera2
import numpy as np
import time

# Load trained YOLO model
model_path = "runs2/detect/train/weights/best.pt"
ourmodel = YOLO(model_path)
class_names = ourmodel.model.names

# Initialize and configure camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(2)

tolerance = 20

def boxes_are_similar(box1, box2, tol):
    return all(abs(a - b) <= tol for a, b in zip(box1, box2))

try:
    while True:
        # Capture frame continuously
        frame = picam2.capture_array()

        # Perform detection
        results = ourmodel(frame)
        result = results[0]

        frame_boxes = [tuple(map(int, box.xyxy[0])) for box in result.boxes]

        similar_boxes_counts = []
        for box in frame_boxes:
            if not any(boxes_are_similar(box, existing_box, tolerance) for existing_box in similar_boxes_counts):
                similar_boxes_counts.append(box)

        box_counts = Counter()
        for box in frame_boxes:
            for similar_box in similar_boxes_counts:
                if boxes_are_similar(box, similar_box, tolerance):
                    box_counts[similar_box] += 1
                    break

        threshold = int(0.8 * 1)  # Each iteration is a single frame
        consistent_boxes = [box for box, count in box_counts.items() if count > threshold]
        centre_boxes = [((x1 + x2)/2, (y1 + y2)/2) for x1, y1, x2, y2 in consistent_boxes]

        # Print detected boxes each iteration clearly
        print("\nConsistent Bounding boxes:")
        for box in consistent_boxes:
            print(box)

        print("Centre coordinates of consistent boxes:")
        for center in centre_boxes:
            print(center)

        with open("/dev/shm/centre_of_boxes.txt", "w") as f:
            for center in centre_boxes:
                f.write(str(center) + "\n")

        # Draw bounding boxes on the current frame
        for x1, y1, x2, y2 in consistent_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Weed", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Save current detection image continuously
        cv2.imwrite("output_image.jpg", frame)

        # Optional display (uncomment if GUI is connected)
        # cv2.imshow('Real-Time Detection', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

except KeyboardInterrupt:
    print("\nExiting due to user interruption...")

finally:
    picam2.stop()
    # cv2.destroyAllWindows()  # Uncomment if using imshow
