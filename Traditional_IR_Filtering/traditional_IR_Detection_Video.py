import cv2
import numpy as np
import multiprocessing as mp
import time
import statistics
from skimage.measure import label, regionprops
from skimage.transform import hough_circle, hough_circle_peaks

# Video Path
video_path = "./IR_Detection/Videos/video_10metres.mp4"
# video_path = "./IR_Detection/Videos/video_15_out.avi"
# video_path = "./IR_Detection/Videos/video_15metres.avi"
# video_path = "./IR_Detection/Videos/video_20metres_out.avi"
cap = cv2.VideoCapture(video_path)

# Multiprocessing Parameters
MAX_WORKERS = mp.cpu_count()
FRAME_BUFFER_SIZE = 30  # Buffer size for processed frames

def process_frame(frame):
    """Processes a single frame, detects circles, and draws one around the most circular object."""
    start_time = time.time()
    
    # Convert to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) / 255.0
    
    # Adaptive thresholding
    binary_image = frame_gray > 0.9
    
    # Remove large non-circular areas
    labeled_img = label(binary_image)
    mask = np.zeros_like(binary_image, dtype=bool)
    
    for region in regionprops(labeled_img):
        if 30 <= region.area <= 500 and region.eccentricity < 0.68:
            mask[labeled_img == region.label] = True
    
    # Apply Gaussian blur to smooth edges
    blurred = cv2.GaussianBlur((mask * 255).astype(np.uint8), (5, 5), 0)
    
    # Detect circles using Hough Transform
    hough_radii = np.arange(15, 200, 5)
    hough_res = hough_circle(blurred, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=5)
    
    # Find the most circular object
    most_circular_idx = None
    lowest_eccentricity = float("inf")
    
    for i, (x, y, r) in enumerate(zip(cx, cy, radii)):
        region_mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.circle(region_mask, (x, y), r, 1, -1)
        labeled_mask = label(region_mask)
        regions = regionprops(labeled_mask)
        
        if regions:
            eccentricity = regions[0].eccentricity
            if eccentricity < lowest_eccentricity:
                lowest_eccentricity = eccentricity
                most_circular_idx = i
    
    # Draw the most circular object
    if most_circular_idx is not None:
        x, y, r = cx[most_circular_idx], cy[most_circular_idx], radii[most_circular_idx]
        cv2.circle(frame_gray, (x, y), r, (0, 0, 255), 2)  # Draw red circle
    
    end_time = time.time()
    return (frame_gray * 255).astype(np.uint8), end_time - start_time

def display_frames(frame_queue):
    """Continuously displays frames from the queue."""
    while True:
        frame_gray = frame_queue.get()
        if frame_gray is None:
            break  # Stop if sentinel value received
        
        cv2.imshow("Detected Target", frame_gray)
        time.sleep(0.1)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cv2.destroyAllWindows()

def main():
    """Reads frames and processes them using multiprocessing with buffering."""
    frame_queue = mp.Queue(maxsize=FRAME_BUFFER_SIZE)
    processing_times = []
    pool = mp.Pool(processes=MAX_WORKERS)
    
    # Start display process
    display_process = mp.Process(target=display_frames, args=(frame_queue,), daemon=True)
    display_process.start()
    
    results = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Submit frame for processing
        result = pool.apply_async(process_frame, (frame,))
        results.append(result)
        
        if len(results) >= FRAME_BUFFER_SIZE:
            for res in results[:FRAME_BUFFER_SIZE]:
                frame_gray, processing_time = res.get()
                frame_queue.put(frame_gray)
                processing_times.append(processing_time)
            results = results[FRAME_BUFFER_SIZE:]
    
    # Process remaining frames
    for res in results:
        frame_gray, processing_time = res.get()
        frame_queue.put(frame_gray)
        processing_times.append(processing_time)
    
    # Stop the display process
    frame_queue.put(None)
    display_process.join()
    
    cap.release()
    pool.close()
    pool.join()
    
    # Compute and print the average frame processing time
    if processing_times:
        avg_time = statistics.mean(processing_times)
        print(f"Average frame processing time: {avg_time:.4f} seconds")

if __name__ == "__main__":
    main()
