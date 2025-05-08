import socket
import threading

IP = '10.42.0.1'
PORT = 5000

exit_event = threading.Event()

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((IP, PORT))
    print(f'Listening on {IP}:{PORT}')

    t = threading.Thread(target=runa, args=(sock,))
    t.start()

    input('Exit? (enter)')
    exit_event.set()

    t.join()

    sock.close()

def runa(socket:socket.socket, ):
    f = open('umuas_kml_file.xml', 'w')
    while True:
        if exit_event.is_set(): break

        data = socket.recv(1024).decode()
        if not data:
            break

        f.write(data)

        print('Received message:')
        print(data)

if __name__ == '__main__':
    main()