def do_ir_detection():
    global hotspot_data, save_state, current_coordinates, print_hotspots, is_valid_index, get_valid_coordinate
    '''
    === IR Detection ===
    Options:
    "add" - Add current GPS position as IR source.
    "rem" - Remove an IR source.
    "set" - Change coordinates of an IR source.
    "list" - List of IR sources added.
    "exit" - Exit IR detection.
    '''
    options = ["add", "rem", "set", "list", "exit"]

    while True:
        print(do_ir_detection.__doc__)
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

