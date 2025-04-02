import numpy as np
import cv2 as cv

######################
# Utility Functions: #
######################

def pixel_distance_from_box_center_to_image_center(img:cv.Mat, boundingBox:cv.typing.Rect) -> tuple[int, int]:
    boxCenter = get_bb_center(boundingBox)
    imgCenter = get_img_center(img)
    return (imgCenter[0]-boxCenter[0], imgCenter[1]-boxCenter[1])

def find_centroid_from_binary_image(img:cv.Mat) -> tuple[int,int]:
    '''Returns: Centroid (x,y) or (-1,-1) if image is zero (black).'''
    M = cv.moments(img)
    centroid = (-1, -1)
    non_zero = cv.findNonZero(img)
    if(non_zero is not None):
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centroid = (cX, cY)
    return centroid

def get_bb_center(boundingBox:cv.typing.Rect) -> tuple[int, int]:
    '''Returns: Center (x,y)'''
    return (boundingBox[0]+boundingBox[2]//2, boundingBox[1]+boundingBox[3]//2)

def get_img_center(img:cv.Mat) -> tuple[int,int]:
    return (img.shape[1]//2, img.shape[0]//2)

###########################
# Functions for the demo: #
###########################

def handleMouse(event, x, y, flags, src_img):
    if(event == cv.EVENT_LBUTTONDOWN):
        src_img = cv.circle(src_img, (x,y), 1, (255, 255, 255), 10)
        createRegion(src_img)

def createRegion(src_img):
    non_zero = cv.findNonZero(src_img)
    bb = cv.boundingRect(non_zero)
    boxCenter = get_bb_center(bb)
    imgCenter = get_img_center(src_img)
    distance = pixel_distance_from_box_center_to_image_center(src_img, bb)
    centroid = find_centroid_from_binary_image(src_img)

    print(f'Centroid: {centroid}, bb center distance to image center: {distance}')
    overlayedImg = cv.cvtColor(src_img, cv.COLOR_GRAY2RGB)
    if(non_zero is not None):
        cv.rectangle(overlayedImg, bb, (255, 255, 255), 1) #Box edges display
        cv.circle(overlayedImg, boxCenter, 1, (0, 0, 255), 5) #Box center display
        cv.circle(overlayedImg, centroid, 1, (0, 255, 255), 5) #Centroid display
    
    cv.circle(overlayedImg, imgCenter, 1, (255, 0, 255), 10) #Image center display

    #Final Image With Overlay
    cv.imshow('image', overlayedImg)

def main():
    #Just use a thresholded image here for src_img. Any pixel that is not black (0,0,0) will be considered for the center calculations
    SOURCE_IMAGE_PATH = "./images/thresholded.png"
    SOURCE_IMAGE = cv.imread(SOURCE_IMAGE_PATH, cv.IMREAD_GRAYSCALE)

    if(SOURCE_IMAGE is None):
        print('Source image was not found.')
        return

    img = SOURCE_IMAGE.copy()

    cv.namedWindow('image')
    cv.setMouseCallback('image', handleMouse, img)
    cv.imshow('image', img)

    cv.waitKey(0)
    cv.destroyAllWindows()

#Perform demo:
if __name__ == '__main__':
    main()