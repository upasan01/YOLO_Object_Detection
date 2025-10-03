import cv2
import numpy as np

# --- 1. Configuration & File Paths ---
# You need to download these three files and place them in the same directory:
# 1. Weights: yolov3.weights (Huge file, download link needed)
# 2. Config: yolov3.cfg 
# 3. Class Names: coco.names

# Recommended Model (YOLOv3-320 is faster than YOLOv3):
WEIGHTS_PATH = "yolov3.weights"  
CONFIG_PATH = "yolov3.cfg"
NAMES_PATH = "coco.names"
INPUT_IMAGE_PATH = "test_image.jpg" # Change this to your image file

# --- 2. Load Model and Configuration ---

def load_yolo():
    """Loads the YOLO model and class names."""
    try:
        # Load the pre-trained neural network from the configuration and weights files
        net = cv2.dnn.readNet(WEIGHTS_PATH, CONFIG_PATH)
    except cv2.error as e:
        print(f"Error loading model files: {e}")
        print("Please ensure yolov3.weights and yolov3.cfg are in the correct path.")
        return None, None

    # Get the names of all layers
    layer_names = net.getLayerNames()
    # Determine the names of the output layers
    unconnected_layers = net.getUnconnectedOutLayers()
    if unconnected_layers.ndim == 1:
        # For single-dimension arrays (common case)
        output_layers = [layer_names[i - 1] for i in unconnected_layers]
    else:
        # For older OpenCV versions/different array formats
        output_layers = [layer_names[i[0] - 1] for i in unconnected_layers]

    # Load all 80 object class names (e.g., person, dog, car)
    with open(NAMES_PATH, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    return net, output_layers, classes

# --- 3. Process Detections ---

def process_detections(net, output_layers, classes, img):
    """
    Runs the detection, draws bounding boxes, and displays the image.
    """
    height, width, channels = img.shape

    # YOLO requires a specific format called a 'blob'
    # 0.00392 is the scale factor (1/255) for normalization
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    
    net.setInput(blob)
    # Run forward pass to get detection results from the output layers
    outs = net.forward(output_layers)

    # Variables to store detection results
    class_ids = []
    confidences = []
    boxes = []
    
    # Process each detection output
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5: # Confidence threshold
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression (NMS) to eliminate redundant, overlapping boxes
    # The result contains the indices of the boxes to keep
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) # conf_threshold=0.5, nms_threshold=0.4

    # --- 4. Drawing Bounding Boxes ---
    
    # Ensure indexes is a flattened array
    if isinstance(indexes, tuple): # Handle case where NMSBoxes returns empty tuple
        indexes = []
    else:
        indexes = indexes.flatten()

    font = cv2.FONT_HERSHEY_PLAIN
    
    # Draw the filtered boxes
    for i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = (0, 255, 0) # Green box

        # Draw box and label
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        text = f"{label} {confidence:.2f}"
        cv2.putText(img, text, (x, y - 10), font, 1.5, color, 2)

    return img

# --- 5. Main Execution ---

def main():
    """Main function to run the YOLO detector."""
    # Load all necessary files
    net, output_layers, classes = load_yolo()
    if net is None:
        return

    # Load input image
    img = cv2.imread(INPUT_IMAGE_PATH)
    if img is None:
        print(f"Error loading image: {INPUT_IMAGE_PATH}")
        print("Please ensure your image file exists.")
        return

    print(f"Running detection on {INPUT_IMAGE_PATH}...")
    
    # Process the image
    detected_img = process_detections(net, output_layers, classes, img)

    # Display result
    cv2.imshow("YOLO Object Detection Result", detected_img)
    cv2.waitKey(0) # Wait indefinitely until a key is pressed
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
