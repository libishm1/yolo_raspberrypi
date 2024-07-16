import torch
import cv2
import numpy as np
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run inference with a TorchScript model.")
parser.add_argument('--model', type=str, required=True, help='Path to the TorchScript model')
parser.add_argument('--image', type=str, required=True, help='Path to the input image')
parser.add_argument('--output', type=str, required=True, help='Path to save the output image')
args = parser.parse_args()

# Check if CUDA is available
device = torch.device('cpu')

# Load the TorchScript model and move it to the device (CPU)
model = torch.jit.load(args.model, map_location=device)
model.eval()

# Load an image using OpenCV
img = cv2.imread(args.image)
assert img is not None, 'Image not found at {}'.format(args.image)

# Preprocess the image
img_resized = cv2.resize(img, (640, 640))
img_transposed = np.transpose(img_resized, (2, 0, 1))  # HWC to CHW
img_expanded = np.expand_dims(img_transposed, axis=0)  # Add batch dimension
img_tensor = torch.from_numpy(img_expanded).float() / 255.0  # Normalize

# Move the tensor to the same device as the model
img_tensor = img_tensor.to(device)

# Run inference
with torch.no_grad():
    results = model(img_tensor)

# Debug: Print the results to understand the structure
print("Results shape:", results[0].shape)
print("Sample detection:", results[0][0].cpu().numpy())

# Define class names (update with your dataset's class names)
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Process results
detections = results[0][0].cpu().numpy()  # Take the first element in the batch

# Set a confidence threshold to filter out low-confidence detections
confidence_threshold = 0.5  # Adjust this value as needed

for detection in detections:
    # Assuming the format is [x, y, w, h, conf, class0_conf, class1_conf, ...]
    x, y, w, h, conf = detection[:5]
    class_scores = detection[5:]
    
    if conf > confidence_threshold:
        cls = np.argmax(class_scores)
        class_conf = class_scores[cls]
        
        # Convert bbox from center format to corner format
        x1 = int((x - w/2) * img.shape[1])
        y1 = int((y - h/2) * img.shape[0])
        x2 = int((x + w/2) * img.shape[1])
        y2 = int((y + h/2) * img.shape[0])
        
        label = f'{class_names[cls]}: {class_conf:.2f}'
        print(f"Detected: Class {class_names[cls]} with confidence {class_conf:.2f} at location {x1},{y1},{x2},{y2}')
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Save and show the image
cv2.imwrite(args.output, img)
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
