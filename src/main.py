import os
import json
import threading
import simplekml
import socket

import argparse 
from pymavlink import mavutil

#Global Variables 
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

hotspot_data = None
mavlink_connection = None
current_coordinates = None 
program_data = {}

#Helper functions 
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def establish_mavlink_connection():
    global program_data, mavlink_connection
    # mavlink_connection = mavutil.mavlink_connection(f'udp:{program_data["udp_address"]}:{program_data["udp_port"]}')
    mavlink_connection = mavutil.mavlink_connection('/dev/ttyTHS1', baud=57600) #FINALY WORKED
    mavlink_connection.wait_heartbeat()
    print("[o] Mavlink connection established")

def do_get_current_coordinates(mavlink_connection):
    # Should be run as a separate thread to constantly update current coordinates
    global current_coordinates

    while True:
        try:
            msg = mavlink_connection.recv_match(blocking=True, timeout=1)
            if msg and msg.get_type() == "GLOBAL_POSITION_INT":
                latitude = msg.lat / 1e7  # Convert to decimal degrees
                longitude = msg.lon / 1e7
                current_coordinates = (latitude, longitude)
        except Exception as e:
            print(f"[x] Error receiving MAVLink message: {e}")

def get_valid_coordinate(prompt, min_val, max_val):
    while True:
        user_input = input(prompt)
        try:
            value = float(user_input)
            if min_val <= value <= max_val:
                return value
            else:
                print(f"[x] Value must be between {min_val} and {max_val}. Try again.")
        except ValueError:
            print("[x] Invalid number. Please enter a numeric value.")

def save_state():
    '''Saves current program state to a file.'''
    global hotspot_data, program_data
    print(f'Writing program state to: {program_data["json_filepath"]}')
    with open(program_data["json_filepath"], 'w') as f:
        json.dump(hotspot_data, f, indent=4)

def is_valid_index(i:int, lst:list):
    return i >= 0 and i < len(lst)

def print_hotspots():
    global hotspot_data
    print("Current hotspots:")
    for i in range(len(hotspot_data["hotspots"])):
        print(f'{i} -> longitude:{hotspot_data["hotspots"][i][0]}  latitude:{hotspot_data["hotspots"][i][1]}')
    pass


# Multithreading Setup 
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def init_coordinates_thread():
    coordinates_thread = threading.Thread(target=do_get_current_coordinates, args=[mavlink_connection], daemon=True)
    coordinates_thread.start()

# Processes
#
#

def do_source_detection():
    global current_coordinates, hotspot_data

    options = ["pos", "desc", "info", "exit"]

    while True:
        print(
'''
=== Fire Source Detection ===
Options:
"pos"  - Set coordinates as the current GPS position.
"desc" - Set description.
"info" - Current source information.
"exit" - Exit source detection.
'''
        )
        choice = input('Choose option: ').strip().lower()

        while choice not in options:
            choice = input('Invalid input! Try again: ').strip().lower()

        if choice == 'exit':
            print('[o] Exiting source detection.')
            save_state()
            print('[o] Program state saved to JSON file.')
            break

        elif choice == 'pos':
            confirm = input('[!] Are you sure you want to update coordinates for the Fire Source? (y/n): ').strip().lower()
            if confirm != 'y':
                continue
            if current_coordinates:
                hotspot_data['source']['coordinates'] = current_coordinates
                save_state()
                print('[o] Fire Source coordinates updated and program state saved.')
            else:
                print('[x] No current coordinates available.')

        elif choice == 'desc':
            current_desc = hotspot_data['source'].get('description', '')
            print(f'Current source description:\n{current_desc}')
            new_desc = input('Change description to (or enter "cancel"):\n').strip()
            if new_desc.lower() == 'cancel':
                continue
            hotspot_data['source']['description'] = new_desc
            save_state()
            print('[o] Fire Source description updated and program state saved.')

        elif choice == 'info':
            desc = hotspot_data['source'].get('description', '[none]')
            coords = hotspot_data['source'].get('coordinates', '[none]')
            print(f'[i] Source description:\n{desc}')
            print(f'[i] Source coordinates:\n{coords}')

def do_ir_detection():
    global hotspot_data, current_coordinates

    options = ["add", "rem", "set", "list", "exit"]

    while True:
        print(
'''
=== IR Detection ===
Options:
"add" - Add current GPS position as IR source.
"rem" - Remove an IR source.
"set" - Change coordinates of an IR source.
"list" - List of IR sources added.
"exit" - Exit IR detection.
'''
        )
        option = input('Choose option: ').lower()

        while option not in options:
            option = input('Invalid input! Try again: ').lower()

        if option == 'exit':
            print('[o] Exiting IR detection.')
            save_state()
            print("[o] Program state saved to JSON file")
            break

        elif option == 'add':
            hotspot_data["hotspots"].append(current_coordinates)
            save_state()
            print("[o] New hotspot added and program state saved to JSON file")

        elif option == 'rem':
            print_hotspots()
            print('[i] Remove a hotspot by its index in the list. To cancel, type "cancel".')

            index_str = input('Choose index: ').lower()
            while (not index_str.isnumeric() and index_str != "cancel") or \
                  (index_str.isnumeric() and not is_valid_index(int(index_str))):
                index_str = input('Invalid input! Try again: ').lower()

            if index_str == 'cancel':
                continue
            else:
                hotspot_index = int(index_str)
                hotspot_data["hotspots"].pop(hotspot_index)
                save_state()
                print("[o] Hotspot removed and program state saved to JSON file")

        elif option == 'set':
            print_hotspots()
            print("Modify Hotspot coordinates by its index in the list. To cancel, type 'cancel'.")

            index_str = input("Choose hotspot: ").lower()
            while (not index_str.isnumeric() and index_str != "cancel") or (index_str.isnumeric() and not is_valid_index(int(index_str))):
                index_str = input('Invalid input! Try again: ').lower()

            if index_str == "cancel":
                continue
            else:
                hotspot_index = int(index_str)
                latitude = get_valid_coordinate("Enter latitude (-90 to 90): ", -90, 90)
                longitude = get_valid_coordinate("Enter longitude (-180 to 180): ", -180, 180)
                hotspot_data["hotspots"][hotspot_index] = (longitude, latitude)
                save_state()
                print("[o] Hotspot data modified and program state saved to JSON file")

        elif option == 'list':
            print_hotspots()

def do_kml_generation():
    global program_data, hotspot_data
    try:
        kml = simplekml.Kml()

        for i, hotspot in enumerate(hotspot_data.get("hotspots", [])):
            kml.newpoint(name=f"Hotspot {i+1}", coords=[hotspot])

        source_coords = hotspot_data.get("source", {}).get("coordinates")
        source_desc = hotspot_data.get("source", {}).get("description", "")

        if source_coords:
            kml.newpoint(name="Source", coords=[source_coords], description=source_desc)

        kml.save(program_data["kml_file_path"])
        print(f'[o] KML file saved to -> {program_data["kml_file_path"]}')

    except Exception as e:
        print(f'[x] Error occurred generating KML file: {e}')

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

# Main Function Logic 
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def main():
    global hotspot_data, program_data
    parser = argparse.ArgumentParser(description='UMUAS Hotspot Detection Program')
 
    # parser.add_argument("-i","--ir",help='initiate IR detection routine', required=False, action='store_true')
    # parser.add_argument("-s","--source",help='initiate Source detection routine', required=False, action='store_true')
    # parser.add_argument('-g',"--generate",help='initiate KML generation', required=False, action="store_true")
    # parser.add_argument('-t',"--transmit", help="transmit kml file", required=False, action="store_true")

    parser.add_argument("-j", "--json", help="json file for storing", required=False, default="state.json")
    parser.add_argument("-kf","--kmlfile", help="file to generate kml data in", default="hotspots.kml")
    parser.add_argument("-ka","--kmlserver",help="kml server address. Defaults to 0.0.0.0",default="0.0.0.0")
    parser.add_argument("-kp","--kmlport",help="kml server port number. Defaults to 5000",default=5000)
    # parser.add_argument("-up","--uport", help="mavlink udp port. Defaults to 14550",default="14550")
    # parser.add_argument("-ua","--uaddress",help="mavlink udp address",default="127.0.0.1")
    
    args = parser.parse_args()
    
    #update program global variables 
    # program_data["udp_port"] = args.uport 
    # program_data["udp_address"] = args.uaddress
    program_data["kml_file_path"] = args.kmlfile
    program_data["kml_server_port"] = args.kmlport 
    program_data["kml_server_address"] = args.kmlserver 
    program_data["json_filepath"] = args.json

    try:
        with open(program_data["json_filepath"], 'r') as f:
            previous_state_data = f.read()
            if len(previous_state_data):
                hotspot_data = json.loads(previous_state_data)
            else:
                hotspot_data = {"hotspots": [], "source": {"description": "", "coordinates": ""}}
    except FileNotFoundError:
        hotspot_data = {"hotspots": [], "source": {"description": "", "coordinates": ""}}

    options = ['1','2','3','4','exit']

    # Run Appropriate Processes
    # establish_mavlink_connection()

    while True:
        print('''
              Options:
              1 - Do IR detection
              2 - Do Source detection
              3 - Do KML generation
              4 - Transit KML
              exit - To exit program.
              ''')
        opt = input('Select option: ').lower()
        while opt not in options:
            opt = input('Invalid input! Try again: ').lower()

        if(opt == '1'):
            # init_coordinates_thread()
            do_ir_detection()
        elif(opt == '2'):
            # init_coordinates_thread()
            do_source_detection()
        elif(opt == '3'):
            do_kml_generation()
        elif(opt == '4'):
            do_kml_transmit()
        else: #exit
            break
            
    save_state()
    print('[o] Program ended.')

if __name__ == '__main__':
    main()

