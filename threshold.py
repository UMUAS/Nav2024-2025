import numpy as np
import cv2 as cv

def nothing(x):
    pass

SOURCE_PATH = "images/easy1.jpg"
SOURCE_IMAGE = cv.imread(SOURCE_PATH, cv.IMREAD_GRAYSCALE) #Do not modify
blurred = None
resized = None
resized_unfiltered = None
img = np.zeros((SOURCE_IMAGE.shape[0], SOURCE_IMAGE.shape[1]), np.uint8)

scale = 1.0
min_scale, max_scale = 0.1, 5.0
current_threshold = 0

#First: blur
#Then: threshold
#Then: resize and display

def on_threshold_change(threshold_value: int):
    """ Thresholds the blurred image """
    global img, current_threshold
    current_threshold = threshold_value

    #NOTE: Would probably want to use cv.THRESH_BINARY instead for the line below
    # _, img = cv.threshold(blurred if blurred is not None else SOURCE_IMAGE, threshold_value, 255, cv.THRESH_TOZERO)
    _, img = cv.threshold(blurred if blurred is not None else SOURCE_IMAGE, threshold_value, 255, cv.THRESH_BINARY)
    show_image(img)

def on_blur_change(blur_amount: int):
    """ Blurs and applies thresholding again"""
    global blurred
    if(blur_amount == 0):
        blurred = None
    else:
        blurred = cv.GaussianBlur(SOURCE_IMAGE, (blur_amount*2+1,blur_amount*2+1), 0)
        
    on_threshold_change(current_threshold)

def handle_mouse(event, x, y, flags, _):
    """ Zoom on ctrl+scroll, set threshold when holding left button """
    global scale
    if(event == cv.EVENT_MOUSEWHEEL and flags & cv.EVENT_FLAG_CTRLKEY):
        if( flags > 0):
            scale *= 1.1
        else:
            scale *= 0.9
        
        scale = max(min_scale, min(max_scale, scale))
        show_image(img)

    if(flags == cv.EVENT_FLAG_LBUTTON):
        cv.setTrackbarPos('Threshold', 'image', resized_unfiltered[y,x])

def show_image(img):
    """ Resize and display the resized image. """
    global resized, resized_unfiltered
    h, w = img.shape[:2]
    resized = cv.resize(img, (int(w*scale), int(h*scale)), interpolation=cv.INTER_LINEAR)
    resized_unfiltered = cv.resize(SOURCE_IMAGE, (int(w*scale), int(h*scale)), interpolation=cv.INTER_LINEAR)
    cv.imshow('image', resized)

#Create UI
cv.namedWindow("image")
cv.createTrackbar("Threshold", "image", 0, 255, on_threshold_change)
cv.createTrackbar("Blur", "image", 0, 5, on_blur_change)
cv.setMouseCallback('image', handle_mouse)

#Initialize
on_blur_change(0)

cv.waitKey(0)
cv.destroyAllWindows()