import os
import json
import threading

import argparse 
from pymavlink import mavutil
from processes.ir_detection import do_ir_detection 
from processes.source_detection import do_source_detection
from processes.kml_generation import do_kml_generation
from processes.kml_transmit import do_kml_transmit
from processes.get_current_coordinates import do_get_current_coordinates

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
    mavlink_connection = mavutil.mavlink_connection(f'udp:{program_data["udp_address"]}:{program_data["udp_port"]}')
    mavlink_connection.wait_heartbeat()
    print("[o] Mavlink connection established")

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



# Main Function Logic 
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def main():
    global hotspot_data, program_data
    parser = argparse.ArgumentParser(description='UMUAS Hotspot Detection Program')
 
    parser.add_argument("-i","--ir",help='initiate IR detection routine', required=False, action='store_true')
    parser.add_argument("-s","--source",help='initiate Source detection routine', required=False, action='store_true')
    parser.add_argument('-g',"--generate",help='initiate KML generation', required=False, action="store_true")
    parser.add_argument('-t',"--transmit", help="transmit kml file", required=False, action="store_true")

    parser.add_argument("-j", "--json", help="json file for storing", required=False, default="state.json")
    parser.add_argument("-kf","--kmlfile", help="file to generate kml data in", default="hotspots.kml")
    parser.add_argument("-ka","--kmlserver",help="kml server address. Defaults to 0.0.0.0",default="0.0.0.0")
    parser.add_argument("-kp","--kmlport",help="kml server port number. Defaults to 5000",default=5000)
    parser.add_argument("-up","--uport", help="mavlink udp port. Defaults to",default="14550")
    parser.add_argument("-ua","--uaddress",help="mavlink udp address",default="127.0.0.1")
    
    args = parser.parse_args()

    #error handling for argparser 
    #only run when 1 intruction is called (i.e program cant perform ir_detection, source detection, ... simultaneously )
    if sum([args.source, args.generate, args.transmit, args.ir]) > 1:
        print("[x] program can't perform more than 1 instruction at a time. exiting ...")
        exit()
    if sum([args.source, args.generate, args.transmit, args.ir]) == 0:
        print('[x] no instruction passed. exiting ...')
        exit()
    
    #update program global variables 
    program_data["udp_port"] = args.uport 
    program_data["udp_address"] = args.uaddress
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

    # Run Appropriate Processes
    establish_mavlink_connection()
    if args.ir:
        init_coordinates_thread()
        do_ir_detection()
    elif args.source:
        init_coordinates_thread()
        do_source_detection()
    elif args.generate:
        do_kml_generation()
    elif args.transmit:
        do_kml_transmit()
        
    save_state()
    print('[o] Program ended.')

if __name__ == '__main__':
    main()

