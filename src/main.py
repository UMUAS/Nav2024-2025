from pymavlink import mavutil

#initlize seperate thread for each task
def init():
    pass

def talk_to_pixhawk():
    # Connect to the Pixhawk over UDP
    connection = mavutil.mavlink_connection('udp:127.0.0.1:14550')

    # Wait for a heartbeat so we know the connection is active
    connection.wait_heartbeat()

def do_triangulation():
    pass

def do_ir_detection():
    pass

def do_source_detection():
    pass

def do_kml_generation():
    pass

def do_transmit_kml():
    pass