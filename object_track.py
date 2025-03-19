import cv2 as cv
import time

tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[4]

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

initBB = None
# fps = None
prev_frame_time = 0 
new_frame_time = 0

cv.namedWindow('preview')
vc = cv.VideoCapture(0)
if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv.imshow('preview', frame)
    rval, frame = vc.read()
    rescale = 0.75
    frame = cv.resize(frame, (int(frame.shape[1]*rescale), int(frame.shape[0]*rescale)))

    if(initBB is not None):
        (success, box) = tracker.update(frame)
        if(success):
            (x, y, w, h) = [int(v) for v in box]
            cv.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time

    cv.putText(frame, str(int(fps)), (7, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv.LINE_AA)

    key = cv.waitKey(5)
    if key == 27: # exit on ESC
        break
    elif key == ord('s'):
        initBB = cv.selectROI('preview', frame, True, False)
        if(initBB is not None):
            tracker.init(frame, initBB)

vc.release()
cv.destroyWindow('preview')