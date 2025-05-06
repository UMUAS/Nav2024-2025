import cv2
import socket
import numpy as np

# UDP Configuration
LISTEN_IP = '0.0.0.0'
LISTEN_PORT = 5005

MAX_PACKET_SIZE = 65507
TIMEOUT_SECONDS = 10

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((LISTEN_IP, LISTEN_PORT))
    print(f"Listening to {LISTEN_IP}:{LISTEN_PORT}")
    
    sock.settimeout(TIMEOUT_SECONDS)

    while True:
        try:
            data, _ = sock.recvfrom(65536)

            # Convert bytes back to numpy array
            np_data = np.frombuffer(data, dtype=np.uint8)

            # Decode JPEG image
            frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

            if frame is not None:
                cv2.imshow("UDP Video Stream", frame)
            else:
                print("No Image!")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except socket.timeout:
            print('Connection timeout! Ending program...')
            break

    sock.close()
    cv2.destroyAllWindows()
    print('Program ended.')

if(__name__ == '__main__'):
    main()