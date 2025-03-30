import numpy as np
import cv2 as cv

#Just use a thresholded image here for src_img. Any pixel that is not black (0,0,0) will be considered for the center calculations
# src_img = cv.imread("images/thresholded.png", cv.IMREAD_GRAYSCALE)
src_img = np.zeros((500, 600, 1), dtype=np.uint8)
img = src_img.copy()

def handleMouse(event, x, y, flags, _):
    global img
    if(event == cv.EVENT_LBUTTONDOWN):
        img = cv.circle(img, (x,y), 1, (255, 255, 255), 10)
        createRegion()
    if(event == cv.EVENT_RBUTTONDOWN):
        img = src_img.copy()
        createRegion()

def createRegion():
    global img

    bounds = [0,0,0,0] #x1,y1,x2,y2
    non_zero = cv.findNonZero(img)
    isEmpty = non_zero is None
    if not isEmpty:
        x,y,w,h = cv.boundingRect(non_zero)
        bounds = [x,y,x+w,y+h]
    
    boxCenter = ((bounds[0]+bounds[2])//2, (bounds[1]+bounds[3])//2) #Center of bounding box
    imgCenter = (img.shape[1]//2, img.shape[0]//2)

    moveX, moveY = (0,0)
    if(imgCenter[0] > boxCenter[0]):
        moveX = 1 #Move Right
    elif(imgCenter[0] <= boxCenter[0]):
        moveX = -1 #Move Left

    if(imgCenter[1] > boxCenter[1]):
        moveY = 1 #Move Up
    elif(imgCenter[1] <= boxCenter[1]):
        moveY = -1 #Move Down
    
    M = cv.moments(img)
    if(not isEmpty):
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centroid = (cX, cY)
    else:
        centroid = (-1, -1) #Not found

    print(f'Box Center: {boxCenter}, Image Center: {imgCenter}, Centroid: {centroid}, moveX: {moveX}, moveY: {moveY}')
    overlayedImg = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    if(not isEmpty):
        cv.rectangle(overlayedImg, (bounds[0], bounds[1]), (bounds[2], bounds[3]), (255, 255, 255), 1) #Box edges display
        cv.circle(overlayedImg, boxCenter, 1, (0, 0, 255), 5) #Box center display
        cv.circle(overlayedImg, centroid, 1, (0, 255, 255), 5) #Centroid display
    
    cv.circle(overlayedImg, imgCenter, 1, (255, 0, 255), 10) #Image center display

    #Final Image With Overlay
    cv.imshow('image', overlayedImg)

#Create UI
cv.namedWindow('image')
cv.setMouseCallback('image', handleMouse)

cv.imshow('image', img)

cv.waitKey(0)
cv.destroyAllWindows()