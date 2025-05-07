import cv2
import socket
import numpy as np
import threading

# UDP Configuration
LISTEN_IP = '0.0.0.0'
LISTEN_PORT_1 = 5005
LISTEN_PORT_2 = 5006

MAX_PACKET_SIZE = 65507
TIMEOUT_SECONDS = 3

def main():
    sock1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    sock1.bind((LISTEN_IP, LISTEN_PORT_1))
    sock2.bind((LISTEN_IP, LISTEN_PORT_2))
    
    print(f"Listening to {LISTEN_IP}:{LISTEN_PORT_1}")
    print(f"Listening to {LISTEN_IP}:{LISTEN_PORT_2}")
    
    sock1.settimeout(TIMEOUT_SECONDS)
    sock2.settimeout(TIMEOUT_SECONDS)

    t1 = threading.Thread(target=view_stream, args=(sock1, f'UDP Port {LISTEN_PORT_1}'))
    t2 = threading.Thread(target=view_stream, args=(sock2, f'UDP Port {LISTEN_PORT_2}'))

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    sock1.close()
    sock2.close()
    cv2.destroyAllWindows()
    print('Program ended.')

def view_stream(sock, window_name):
    while True:
        try:
            data, _ = sock.recvfrom(65536)

            # Convert bytes back to numpy array
            np_data = np.frombuffer(data, dtype=np.uint8)

            # Decode JPEG image
            frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

            if frame is not None:
                cv2.imshow(window_name, frame)
            else:
                print("No Image!")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except socket.timeout:
            print('Connection timeout! Ending program...')
            break

if(__name__ == '__main__'):
    main()