import cv2
import socket
import numpy as np
import threading

CLIENT_IP = '127.0.0.1'  # Take first argument when program is run
CLIENT_PORT_0 = 5005
CLIENT_PORT_1 = 5006

MAX_PACKET_SIZE = 65507

stop_event = threading.Event()

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0
    ):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def send_camera(socket:socket.socket, addr, cam_index, frame_processing:function=None):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f'Camera "{cam_index}" is not open.')
        cap.release()
        return
    
    while cap.isOpened():
        if stop_event.is_set():
            break

        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_processing is not None:
            frame = frame_processing(frame)

        _, frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        
        data = frame.tobytes()

        if len(data) > MAX_PACKET_SIZE:
            print(f'Camera {cam_index}: Frame too large for packet!')
            break

        socket.sendto(data, addr)
    
    cap.release()

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)    
    print('Socket established.')

    t1 = threading.Thread(target=send_camera, args=(sock, (CLIENT_IP, CLIENT_PORT_0), gstreamer_pipeline(sensor_id=0), None))
    t2 = threading.Thread(target=send_camera, args=(sock, (CLIENT_IP, CLIENT_PORT_1), gstreamer_pipeline(sensor_id=1), None))

    t1.start()
    t2.start()

    input('Exit program? (enter)')
    stop_event.set()
    
    t1.join()
    t2.join()
    sock.close()

    print('Program ended.')


if(__name__ == '__main__'):
    main()