'''
For usage download models by following links
For GOTURN:
    goturn.prototxt and goturn.caffemodel: https://github.com/opencv/opencv_extra/tree/c4219d5eb3105ed8e634278fad312a1a8d2c182d/testdata/tracking
For DaSiamRPN:
    network:     https://www.dropbox.com/s/rr1lk9355vzolqv/dasiamrpn_model.onnx?dl=0
    kernel_r1:   https://www.dropbox.com/s/999cqx5zrfi7w4p/dasiamrpn_kernel_r1.onnx?dl=0
    kernel_cls1: https://www.dropbox.com/s/qvmtszx5h339a0w/dasiamrpn_kernel_cls1.onnx?dl=0
For NanoTrack:
    nanotrack_backbone: https://github.com/HonglinChu/SiamTrackers/blob/master/NanoTrack/models/nanotrackv2/nanotrack_backbone_sim.onnx
    nanotrack_headneck: https://github.com/HonglinChu/SiamTrackers/blob/master/NanoTrack/models/nanotrackv2/nanotrack_head_sim.onnx

USAGE:
    tracker.py [-h] [--tracker_algo TRACKER_ALGO (boosting, mil, kcf, tld, medianflow, mosse, csrt, goturn, dasiamrpn, nanotrack, vittrack)]
                    [--goturn GOTURN_PROTOTXT]
                    [--goturn_model GOTURN_MODEL]
                    [--dasiamrpn_net DASIAMRPN_NET]
                    [--dasiamrpn_kernel_r1 DASIAMRPN_KERNEL_R1]
                    [--dasiamrpn_kernel_cls1 DASIAMRPN_KERNEL_CLS1]
                    [--nanotrack_backbone NANOTRACK_BACKBONE]
                    [--nanotrack_headneck NANOTRACK_TARGET]
                    [--vittrack_net VITTRACK_MODEL]
                    [--vittrack_net VITTRACK_MODEL]
                    [--tracking_score_threshold TRACKING SCORE THRESHOLD FOR ONLY VITTRACK]
                    [--backend CHOOSE ONE OF COMPUTATION BACKEND]
                    [--target CHOOSE ONE OF COMPUTATION TARGET]
'''

import cv2 as cv
import time
import sys
import argparse

TRACKER_TYPES = ['BOOSTING', 'MIL','KCF', 'TLD', 'MedianFlow', 'MOSSE', 'CSRT', 'GOTURN', 'DaSiamRPN', 'nanotrack', 'VITTrack']
RESOLUTION_SCALE = 1
WINDOW_NAME = 'Preview'
DEFAULT_TRACKER_TYPE = 'DaSiamRPN'

backends = (cv.dnn.DNN_BACKEND_DEFAULT, cv.dnn.DNN_BACKEND_HALIDE, cv.dnn.DNN_BACKEND_INFERENCE_ENGINE, cv.dnn.DNN_BACKEND_OPENCV,
            cv.dnn.DNN_BACKEND_VKCOM, cv.dnn.DNN_BACKEND_CUDA)
targets = (cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_TARGET_OPENCL, cv.dnn.DNN_TARGET_OPENCL_FP16, cv.dnn.DNN_TARGET_MYRIAD,
           cv.dnn.DNN_TARGET_VULKAN, cv.dnn.DNN_TARGET_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16)

class App(object):
    def __init__(self, args):
        self.args = args
        self.trackerAlgorithm = args.tracker_algo
        self.tracker = self.createTracker()
        self.bounding_box = None #This is the bounding box.
    
    def createTracker(self):
        algo = self.trackerAlgorithm.lower()
        if algo == 'boosting':
            tracker = cv.legacy.TrackerBoosting.create()
        elif algo == 'mil':
            tracker = cv.TrackerMIL.create()
        elif algo == 'kcf':
            tracker = cv.TrackerKCF.create()
        elif algo == 'tld':
            tracker = cv.legacy.TrackerTLD.create()
        elif algo == 'medianflow':
            tracker = cv.legacy.TrackerMedianFlow.create()
        elif algo == 'mosse':
            tracker = cv.legacy.TrackerMOSSE.create()
        elif algo == 'csrt':
            tracker = cv.TrackerCSRT.create()
        elif algo == 'goturn':
            params = cv.TrackerGOTURN_Params()
            params.modelTxt = self.args.goturn
            params.modelBin = self.args.goturn_model
            tracker = cv.TrackerGOTURN.create(params)
        elif algo == 'dasiamrpn':
            params = cv.TrackerDaSiamRPN_Params()
            params.model = self.args.dasiamrpn_net
            params.kernel_cls1 = self.args.dasiamrpn_kernel_cls1
            params.kernel_r1 = self.args.dasiamrpn_kernel_r1
            params.backend = self.args.backend
            params.target = self.args.target
            tracker = cv.TrackerDaSiamRPN.create(params)
        elif algo == 'nanotrack':
            params = cv.TrackerNano_Params()
            params.backbone = self.args.nanotrack_backbone
            params.neckhead = self.args.nanotrack_headneck
            params.backend = self.args.backend
            params.target = self.args.target
            tracker = cv.TrackerNano.create(params)
        elif algo == 'vittrack':
            params = cv.TrackerVit_Params()
            params.net = self.args.vittrack_net
            params.tracking_score_threshold = self.args.tracking_score_threshold
            params.backend = self.args.backend
            params.target = self.args.target
            tracker = cv.TrackerVit.create(params)
        else:
            sys.exit(f"Tracker {self.trackerAlgorithm} is not recognized. Valid algorithms: {TRACKER_TYPES}")
        return tracker

    def initializeTracker(self, image):
        while True:
            print('==> Select object ROI for tracker ...')
            self.bounding_box = cv.selectROI(WINDOW_NAME, image)
            print('ROI: {}'.format(self.bounding_box))
            if self.bounding_box[2] <= 0 or self.bounding_box[3] <= 0:
                sys.exit("ROI selection cancelled. Exiting...")

            try:
                self.tracker.init(image, self.bounding_box)
            except Exception as e:
                print('Unable to initialize tracker with requested bounding box. Is there any object?')
                print(e)
                print('Try again ...')
                continue

            return
    
    def run(self) -> any:
        self.running = True

        video_capture = cv.VideoCapture(0)

        if not video_capture.isOpened(): # try to get the first frame
            print('Error: Unable to access the camera.')
            return
        
        rval, frame = video_capture.read()
        if not rval:
            print("Failed to read first frame.")
            return
        
        # cv.namedWindow(WINDOW_NAME)
        
        prev_frame_time = 0
        new_frame_time = 0

        while self.running:
            rval, frame = video_capture.read()
            if not rval:
                print('Failed to read frame.')
                break
            
            frame = self.processImage(frame)
            
            success = False

            # key = cv.waitKey(5)
            # if key == 27: # exit on ESC
            #     break
            # elif key == ord('s'): #Enable tracking.
                
            #     if(self.bounding_box is not None):
            #         self.bounding_box = None
            #         self.tracker = self.createTracker() #Reset tracker
            #     else:
            #         self.initializeTracker(frame)

            # if self.bounding_box is not None:
            #     success, box = self.tracker.update(frame)
            #     self.bounding_box = box
            #     if success:
            #         x, y, w, h = [int(v) for v in box]
            #         cv.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)

            #FPS:
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time

            #Apply overlays
            cv.putText(frame, f"Tracker Algorithm ('s' to toggle tracking): {self.trackerAlgorithm}", (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (10, 255, 10), 1, cv.LINE_AA)
            cv.putText(frame, f"{int(fps)} FPS", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (10, 255, 10), 1, cv.LINE_AA)
            if not success:
                cv.putText(frame, f"Lost object!", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 200), 1, cv.LINE_AA)

            # cv.imshow(WINDOW_NAME, frame)
            return frame

        print('Done.')
        video_capture.release()
    
    def interrupt(self):
        ...
        self.running = False

    def processImage(self, image) -> any:
        new_image = cv.resize(image, (int(image.shape[1]*RESOLUTION_SCALE), int(image.shape[0]*RESOLUTION_SCALE)))
        return new_image

    def get_bounding_box(self):
        return self.bounding_box

def setupArgs() -> any:
    parser = argparse.ArgumentParser(description="Run tracker")
    # parser.add_argument("--input", type=str, default="vtest.avi", help="Path to video source")
    parser.add_argument("--tracker_algo", type=str, default=DEFAULT_TRACKER_TYPE, help="One of available tracking algorithms: mil, goturn, dasiamrpn, nanotrack, vittrack")
    parser.add_argument("--goturn", type=str, default="models/goturn.prototxt", help="Path to GOTURN architecture")
    parser.add_argument("--goturn_model", type=str, default="models/goturn.caffemodel", help="Path to GOTERN model")
    parser.add_argument("--dasiamrpn_net", type=str, default="models/dasiamrpn_model.onnx", help="Path to onnx model of DaSiamRPN net")
    parser.add_argument("--dasiamrpn_kernel_r1", type=str, default="models/dasiamrpn_kernel_r1.onnx", help="Path to onnx model of DaSiamRPN kernel_r1")
    parser.add_argument("--dasiamrpn_kernel_cls1", type=str, default="models/dasiamrpn_kernel_cls1.onnx", help="Path to onnx model of DaSiamRPN kernel_cls1")
    parser.add_argument("--nanotrack_backbone", type=str, default="models/nanotrack_backbone_sim.onnx", help="Path to onnx model of NanoTrack backBone")
    parser.add_argument("--nanotrack_headneck", type=str, default="models/nanotrack_head_sim.onnx", help="Path to onnx model of NanoTrack headNeck")
    parser.add_argument("--vittrack_net", type=str, default="models/vitTracker.onnx", help="Path to onnx model of  vittrack")
    parser.add_argument('--tracking_score_threshold', type=float,  help="Tracking score threshold. If a bbox of score >= 0.3, it is considered as found ")
    parser.add_argument('--backend', choices=backends, default=cv.dnn.DNN_BACKEND_DEFAULT, type=int,
                help="Choose one of computation backends: "
                        "%d: automatically (by default), "
                        "%d: Halide language (http://halide-lang.org/), "
                        "%d: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                        "%d: OpenCV implementation, "
                        "%d: VKCOM, "
                        "%d: CUDA"% backends)
    #default: DNN_BACKEND_DEFAULT
    parser.add_argument("--target", choices=targets, default=cv.dnn.DNN_TARGET_CPU, type=int,
                help="Choose one of target computation devices: "
                        '%d: CPU target (by default), '
                        '%d: OpenCL, '
                        '%d: OpenCL fp16 (half-float precision), '
                        '%d: VPU, '
                        '%d: VULKAN, '
                        '%d: CUDA, '
                        '%d: CUDA fp16 (half-float preprocess)'% targets)
    #default: DNN_TARGET_CPU

    return parser.parse_args()

if __name__ == "__main__":
    print(__doc__)
    args = setupArgs()
    
    App(args).run()
    cv.destroyWindow(WINDOW_NAME)