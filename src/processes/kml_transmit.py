import socket
import os

def do_kml_transmit():
    global program_data

    if not os.path.exists(program_data["kml_file_path"]):
        print(f'[x] KML file not found at {program_data["kml_file_path"]}')
        return

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((program_data["kml_server_address"], int(program_data["kml_server_port"])))
            sock.listen(1)
            sock.settimeout(60)
            print(f'[i] KML server listening on {program_data["kml_server_address"]}:{program_data["kml_server_port"]}...')

            while True:
                try:
                    conn, addr = sock.accept()
                    with conn:
                        print(f'[o] Connected to receiver at {addr}')
                        with open(program_data["kml_file_path"], "rb") as f:
                            while True:
                                chunk = f.read(1024)
                                if not chunk:
                                    break
                                conn.sendall(chunk)
                        print('[o] KML file sent successfully')
                except socket.timeout:
                    print('[x] No connections within timeout period.')

                user_input = input('[?] Keep server alive for another connection? (y/n): ').strip().lower()
                if user_input != 'y':
                    print('[i] Shutting down KML server.')
                    break

    except Exception as e:
        print(f'[x] Failed to start/send KML file: {e}')