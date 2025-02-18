import cv2
from ultralytics import YOLO
from collections import Counter

# Load the trained YOLO model
model_path = "runs2/detect/train/weights/best.pt"
ourmodel = YOLO(model_path)

# Get class names from the model
class_names = ourmodel.model.names

# Open the camera
cap = cv2.VideoCapture(0)  # 0 is the default camera

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Define tolerance for bounding box matching
tolerance = 20  

def boxes_are_similar(box1, box2, tol):
    """Check if two bounding boxes are similar within a given tolerance."""
    return (abs(box1[0] - box2[0]) <= tol and abs(box1[1] - box2[1]) <= tol and
            abs(box1[2] - box2[2]) <= tol and abs(box1[3] - box2[3]) <= tol)

# Variable to store the selected frame
selected_frame = None

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Display the resulting frame
    cv2.imshow('Camera Feed', frame)

    # Wait for the Enter key to be pressed
    key = cv2.waitKey(1) & 0xFF
    if key == 13:  # Enter key
        selected_frame = frame.copy()
        print("Frame selected for detection.")
        break
    elif key == 27:  # Esc key to exit
        print("Exiting...")
        break

# Release the camera after capturing frames
cap.release()
cv2.destroyAllWindows()

if selected_frame is not None:
    # Perform detection on the selected frame
    results = ourmodel(selected_frame)  # list of 1 Results object
    result = results[0]

    # Collect bounding box coordinates
    frame_boxes = []
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get the coordinates
        frame_boxes.append((x1, y1, x2, y2))

    # Count occurrences of similar boxes
    similar_boxes_counts = []

    # Compare each box against all previously found boxes
    for box in frame_boxes:
        found_similar = False
        for similar_box in similar_boxes_counts:
            if boxes_are_similar(box, similar_box, tolerance):
                found_similar = True
                break
        if not found_similar:
            similar_boxes_counts.append(box)

    # Flatten the list of bounding boxes and count occurrences
    flat_boxes = frame_boxes
    box_counts = Counter()

    # Count occurrences allowing for tolerance
    for box in flat_boxes:
        for similar_box in similar_boxes_counts:
            if boxes_are_similar(box, similar_box, tolerance):
                box_counts[similar_box] += 1
                break

    # Calculate the threshold for 80% occurrence
    threshold = int(0.8 * 1)  # Only one frame selected

    # Filter boxes that occurred more than 80% of the time
    consistent_boxes = [box for box, count in box_counts.items() if count > threshold]

    # Calculate the center of the consistent boxes
    centre_boxes = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in consistent_boxes]

    # Print the list of consistent bounding boxes
    print("Bounding boxes that occurred more than 80% of the times:")
    for box in consistent_boxes:
        print(box)

    print("Centre of the consistent boxes:")
    for box in centre_boxes:
        print(box)
        
    # Writing center of the boxes to a file in RAM
    # (Using /dev/shm, which is a tmpfs filesystem on Linux)
    with open("/dev/shm/centre_of_boxes.txt", "w") as f:
        for box in centre_boxes:
            f.write(str(box) + "\n")

    # Optional: Draw the consistent boxes on the selected frame
    for box in consistent_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(selected_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw in green
        cv2.putText(selected_frame, "Weed", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save or display the output image
    output_image_path = "output_image.jpg"
    cv2.imwrite(output_image_path, selected_frame)
    print(f"Output image saved to: {output_image_path}")

    # Display the image
    cv2.imshow('Image with Consistent Boxes', selected_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()