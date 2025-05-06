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