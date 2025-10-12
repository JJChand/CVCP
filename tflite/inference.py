import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
import time

# Quantized Model with quantized weights (int8) and float inputs/outputs (float32)
model_path = './content/best_int8.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Obtain the height and width of the corresponding image from the input tensor
image_height = input_details[0]['shape'][1] # 640
image_width = input_details[0]['shape'][2] # 640

# Image Preparation
image_path = "../datasets_yolocp/weed-crop-aerial-2/test/images/32242_jpg.rf.2d789f1b1a2dd8fa7d2db7f67fcb784c.jpg" 
image = Image.open(image_path).convert("RGB")
image_resized = image.resize((image_width, image_height))

# Input Preprocessing
image_np = np.array(image_resized)
image_np = np.true_divide(image_np, 255, dtype=np.float32) 
image_np = image_np[np.newaxis, :]

# inference
interpreter.set_tensor(input_details[0]['index'], image_np)
start = time.time()
interpreter.invoke()
# print(f'run time：{time.time() - start}s')

# Obtaining output results
output = interpreter.get_tensor(output_details[0]['index'])
output = output[0]
output = output.T

boxes_xywh = output[..., :4] # Coordinates of bounding box (xywh)
scores = np.max(output[..., 5:], axis=1) # Get score value from most likely class
classes = np.argmax(output[..., 5:], axis=1) # Get the most probable class's index

# Threshold Setting
conf_threshold = 0.3
iou_threshold = 0.25

# Post-processing: Non-Maximum Suppression (NMS) to filter overlapped boxes
x, y, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
x1 = x - w / 2
y1 = y - h / 2
x2 = x + w / 2
y2 = y + h / 2

indices = tf.image.non_max_suppression(
    boxes=tf.stack([y1, x1, y2, x2], axis=1),
    scores=scores,
    max_output_size=50,
    iou_threshold=iou_threshold,
    score_threshold=conf_threshold
).numpy()

# Decode and Draw Bounding Boxes
draw = ImageDraw.Draw(image_resized)
for i in indices:
    box = boxes_xywh[i]
    score = scores[i]
    cls = classes[i]
    x_center, y_center, width, height = box
    x1 = int((x_center - width / 2))
    y1 = int((y_center - height / 2))
    x2 = int((x_center + width / 2))
    y2 = int((y_center + height / 2))
    
    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    text = f"Class: {cls}, Score: {score:.2f}"
    draw.text((x1, y1), text, fill="red")

print("Total boxes:", len(indices))
print(f'Run time：{time.time() - start}s')

# Saving Images
image_resized.save(f"./content/output.jpg")
print("Output image saved as output.jpg")