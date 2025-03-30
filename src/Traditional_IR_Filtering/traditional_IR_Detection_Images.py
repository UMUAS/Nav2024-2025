import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.measure import label, regionprops
from skimage.transform import hough_circle, hough_circle_peaks

# Load images in grayscale using OpenCV (faster)
image_paths = [
    "./Images/target_15m_dark_conditions.jpg",
    "./Images/target_10m.jpg",
    "./Images/target_20m_light_to_dark_conditions.jpg",
    "./Images/target_15m_light_conditions.jpg",
]
images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) / 255.0 for path in image_paths]

# Vectorized Thresholding
binary_images = [img > 0.9 for img in images]

# Remove large areas that are not circular
filtered_images = []
for binary_image in binary_images:
    labeled_img = label(binary_image)

    # Filter based on area and shape
    mask = np.zeros_like(binary_image, dtype=bool)
    for region in regionprops(labeled_img):
        if 30 <= region.area <= 500 and region.eccentricity < 0.68:
            mask[labeled_img == region.label] = True

    filtered_images.append(mask)

# Apply Gaussian Blur to smooth the filtered images
blurred_images = [cv2.GaussianBlur((img * 255).astype(np.uint8), (5, 5), 0) for img in filtered_images]

# Optimize Hough Transform parameters
hough_radii = np.arange(15, 200, 5)  # Reduce step size for efficiency

for i, (original, binary, blurred) in enumerate(zip(images, binary_images, blurred_images)):
    hough_res = hough_circle(blurred, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=5)  # Detect multiple circles

    # Find the most circular object (lowest eccentricity)
    most_circular_idx = None
    lowest_eccentricity = float("inf")

    for j, (x, y, r) in enumerate(zip(cx, cy, radii)):
        region_mask = np.zeros_like(binary, dtype=np.uint8)  # Convert to uint8
        cv2.circle(region_mask, (x, y), r, 1, -1)  # Create a filled circle mask
        labeled_mask = label(region_mask)
        regions = regionprops(labeled_mask)
        
        if regions:
            eccentricity = regions[0].eccentricity
            if eccentricity < lowest_eccentricity:  # Lower eccentricity means more circular
                lowest_eccentricity = eccentricity
                most_circular_idx = j

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Display original image with detected circles
    axes[0].imshow(binary, cmap='gray')
    axes[0].set_title("Threshold Image")
    axes[0].axis('off')


    # Display thresholded image
    axes[1].imshow(blurred, cmap='gray')
    axes[1].set_title("Blurred Image")
    axes[1].axis('off')
    
    # Display detected object after Hough Transform
    axes[2].imshow(original, cmap='gray')
    axes[2].set_title("Most Circular Object Detected")
    axes[2].axis('off')

    if most_circular_idx is not None:
        x, y, r = cx[most_circular_idx], cy[most_circular_idx], radii[most_circular_idx]
        detected_circle = plt.Circle((x, y), r, color='b', fill=False, linewidth=1)  # Red highlight
        
        # Add the same circle to both images
        axes[1].add_patch(detected_circle)
        axes[2].add_patch(plt.Circle((x, y), r, color='r', fill=False, linewidth=1))  # Clone the circle


    plt.show()
