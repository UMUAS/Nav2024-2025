def do_source_detection():
    global current_coordinates, hotspot_data, save_state

    '''
    === Fire Source Detection ===
    Options:
    "pos"  - Set coordinates as the current GPS position.
    "desc" - Set description.
    "info" - Current source information.
    "exit" - Exit source detection.
    '''
    options = ["pos", "desc", "info", "exit"]

    while True:
        print(do_source_detection.__doc__)
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
