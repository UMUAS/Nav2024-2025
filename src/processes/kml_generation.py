import simplekml

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