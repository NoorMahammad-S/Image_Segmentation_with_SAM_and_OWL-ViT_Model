import cv2
import numpy as np
import matplotlib.pyplot as plt

# Install and import Meta AI's SAM and OWL-ViT models
# Make sure to replace these with the actual import statements for SAM and OWL-ViT
from sam import SAM
from owl_vit import OWLViT

# Function for image preprocessing
def preprocess_image(image, target_size=(224, 224), chw_format=True):
    # Resize the image to the required input size of the model
    preprocessed_image = cv2.resize(image, target_size)

    # Normalize the image values to be in the range [0, 1]
    preprocessed_image = preprocessed_image.astype(np.float32) / 255.0

    # Ensure the image is in the format expected by the model
    if chw_format:
        preprocessed_image = np.transpose(preprocessed_image, (2, 0, 1))

    return preprocessed_image

# Function for post-processing segmentation mask
def postprocess_segmentation_mask(segmentation_mask):
    # Threshold the segmentation mask to obtain binary mask
    _, binary_mask = cv2.threshold(segmentation_mask, 0.5, 1, cv2.THRESH_BINARY)

    # Apply morphological operations for smoothing and refining the mask
    kernel = np.ones((5, 5), np.uint8)
    postprocessed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    postprocessed_mask = cv2.morphologyEx(postprocessed_mask, cv2.MORPH_OPEN, kernel)

    return postprocessed_mask

# Function for post-processing object detection results
def postprocess_object_detection(results, confidence_threshold=0.5, nms_threshold=0.3):
    # Filter results based on confidence threshold
    filtered_results = [result for result in results if result['confidence'] > confidence_threshold]

    # Sort results by confidence in descending order
    sorted_results = sorted(filtered_results, key=lambda x: x['confidence'], reverse=True)

    # Apply non-maximum suppression (NMS)
    final_results = []
    for result in sorted_results:
        # Check if the result overlaps significantly with any previously selected result
        overlap = False
        for final_result in final_results:
            iou = calculate_intersection_over_union(result['bbox'], final_result['bbox'])
            if iou > nms_threshold:
                overlap = True
                break

        # If there is no significant overlap, add the result to the final results
        if not overlap:
            final_results.append(result)

    return final_results

def calculate_intersection_over_union(box1, box2):
    # Calculate the intersection over union (IoU) of two bounding boxes
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

# Load the image
image_path = 'path/to/your/image.jpg'
image = cv2.imread(image_path)

# Preprocess the image
preprocessed_image = preprocess_image(image)

# Perform image segmentation using SAM
sam_model = SAM()  # Initialize SAM model
segmentation_mask = sam_model.predict(preprocessed_image)

# Post-process the segmentation mask
postprocessed_mask = postprocess_segmentation_mask(segmentation_mask)

# Perform object detection using OWL-ViT
owlvit_model = OWLViT()  # Initialize OWL-ViT model
object_detection_results = owlvit_model.predict(preprocessed_image)

# Post-process the object detection results
postprocessed_results = postprocess_object_detection(object_detection_results)

# Visualize the results
plt.figure(figsize=(12, 6))

# Original Image
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

# Segmentation Mask
plt.subplot(1, 3, 2)
plt.imshow(postprocessed_mask, cmap='gray')
plt.title('Segmentation Mask')

# Visualize post-processed object detection results on the original image
plt.subplot(1, 3, 3)
# Create a copy of the original image for visualization
vis_image = image.copy()
for result in postprocessed_results:
    x, y, w, h = result['bbox']
    label = result['label']

    cv2.rectangle(vis_image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
    cv2.putText(vis_image, label, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
plt.title('Object Detection Results')

plt.show()
