import cv2
from ultralytics import YOLO
from collections import Counter
import numpy as np
import time

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)

def find_centre(box):
    centre_x = (box[0]+box[2])/2
    centre_y = (box[1]+box[3])/2
    return (centre_x, centre_y)

# Load the trained YOLO model
model_path = "runs2/detect/train/weights/best.pt"
ourmodel = YOLO(model_path)

# Get class names from the model
class_names = ourmodel.model.names

# Load a single image
image_path = "datasets_yolocp/weed-crop-aerial-2/test/images/32242_jpg.rf.2d789f1b1a2dd8fa7d2db7f67fcb784c.jpg"  # Change this to your image file path
frame = cv2.imread(image_path)

start = time.time()

if frame is None:
    print("Error: Could not read image.")
    exit()

# Number of frames to simulate (for testing purposes)
num_frames = 10  # Simulate capturing the same image multiple times
all_boxes = []

# Simulate capturing the same image multiple times
for _ in range(num_frames):
    # Run inference on the frame
    results = ourmodel(frame)  # list of 1 Results object
    result = results[0]

    # Collect bounding box coordinates
    frame_boxes = []
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get the coordinates
        frame_boxes.append((x1, y1, x2, y2))

    # Append the boxes detected in this frame to all_boxes
    all_boxes.append(frame_boxes)

# Flatten the list of bounding boxes and count occurrences
flat_boxes = [box for sublist in all_boxes for box in sublist]
box_counts = Counter(flat_boxes)

# Calculate the threshold for 80% occurrence
threshold = int(0.8 * num_frames)

# Filter boxes that occurred more than 80% of the time
consistent_boxes = [box for box, count in box_counts.items() if count > threshold]

centre_boxes = [find_centre(box) for box in consistent_boxes]

polar_boxes = [cart2pol(box[0],box[1]) for box in centre_boxes]

# Print the list of consistent bounding boxes
print("Bounding boxes that occurred more than 80% of the times:")
for box in consistent_boxes:
    print(box)

print("centre of the consistent boxes:")
for box in centre_boxes:
    print(box)

print("poalar coordinate of the centre of the boxes:")
for box in polar_boxes:
    print(box)


# Draw the consistent boxes on the image
for box in consistent_boxes:
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw in green
    x,y = find_centre(box)
    rho,phi = cart2pol(x,y)

    cv2.putText(frame,"("+ "{:.4g}".format(rho) + "," + "{:.3g}".format(phi)+ ")", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Save the output image
output_image_path = "output_image2.jpg"
cv2.imwrite(output_image_path, frame)
print(f"Output image saved to: {output_image_path}")

# display the image
cv2.imshow('Image with Consistent Boxes', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(f"time taken: {time.time()-start}")