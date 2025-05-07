import socket
import threading

IP = '127.0.0.1'
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

def runa(socket:socket.socket):
    while True:
        if exit_event.is_set(): break

        data = socket.recv(1024).decode()
        if not data:
            break

        print('Received message:')
        print(data)

if __name__ == '__main__':
    main()