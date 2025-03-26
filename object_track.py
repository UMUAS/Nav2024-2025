import cv2 as cv
import time

TRACKER_TYPES = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
RESOLUTION_SCALE = 0.75
WINDOW_NAME = 'preview'
TRACKER_TYPE = TRACKER_TYPES[4]

bounding_box = None #This is the bounding box.

def main():
    """
    Tracks a single object from live feed and updates bounding_box.
    To determine what to track, press 's' and then hold and drag to define a region and press 'enter'.
    To stop tracking (or to track again) press 's' again.
    """
    global bounding_box

    tracker = create_tracker(TRACKER_TYPE)

    cv.namedWindow(WINDOW_NAME)

    video_capture = cv.VideoCapture(0)
    if video_capture.isOpened(): # try to get the first frame
        rval, frame = video_capture.read()
    else:
        rval = False
    
    prev_frame_time = 0
    new_frame_time = 0

    while rval:
        cv.imshow(WINDOW_NAME, frame)
        rval, frame = video_capture.read()
        frame = cv.resize(frame, (int(frame.shape[1]*RESOLUTION_SCALE), int(frame.shape[0]*RESOLUTION_SCALE)))

        if(bounding_box is not None):
            (success, box) = tracker.update(frame)
            if(success):
                (x, y, w, h) = [int(v) for v in box]
                cv.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time

        cv.putText(frame, str(int(fps)), (7, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 3, cv.LINE_AA)

        key = cv.waitKey(5)
        if key == 27: # exit on ESC
            break
        elif key == ord('s'):
            if(bounding_box is not None):
                bounding_box = None #Toggle off tracking.
            else:
                bounding_box = cv.selectROI(WINDOW_NAME, frame, showCrosshair=False, fromCenter=False)
                if(bounding_box is not None):
                    tracker.init(frame, bounding_box)

    video_capture.release()
    cv.destroyWindow(WINDOW_NAME)

def create_tracker(tracker_type):
    tracker = None
    if tracker_type == 'BOOSTING':
        tracker = cv.legacy.TrackerBoosting.create()
    if tracker_type == 'MIL':
        tracker = cv.TrackerMIL.create()
    if tracker_type == 'KCF':
        tracker = cv.TrackerKCF.create() 
    if tracker_type == 'TLD':
        tracker = cv.legacy.TrackerTLD.create() 
    if tracker_type == 'MEDIANFLOW':
        tracker = cv.legacy.TrackerMedianFlow.create() 
    if tracker_type == 'GOTURN':
        tracker = cv.TrackerGOTURN.create()
    if tracker_type == 'MOSSE':
        tracker = cv.legacy.TrackerMOSSE.create()
    if tracker_type == "CSRT":
        tracker = cv.TrackerCSRT.create()
    return tracker

if __name__ == "__main__":
    main()