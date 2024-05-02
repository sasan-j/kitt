def calculate_route():
    api_key = "api_key"
    origin = "49.631997,6.171029"
    destination = "49.586745,6.140002"

    url = f"https://api.tomtom.com/routing/1/calculateRoute/{origin}:{destination}/json?key={api_key}"
    response = requests.get(url)
    data = response.json()

    lats = []
    lons = []

    for point in data['routes'][0]['legs'][0]['points']:
        lats.append(point['latitude'])
        lons.append(point['longitude'])
    # fig = px.line_geo(lat=lats, lon=lons)
    # fig.update_geos(fitbounds="locations")

    fig = px.line_mapbox(lat=lats, lon=lons, zoom=12, height=600)

    fig.update_layout(mapbox_style="open-street-map", mapbox_zoom=12, mapbox_center_lat=lats[0], mapbox_center_lon=lons[0])
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    return fig


def find_route(lat_depart="0", lon_depart="0", city_depart="", address_destination="", depart_time ="", **kwargs):
    """
    Return the distance and the estimated time to go to a specific destination from the current place, at a specified depart time.
    :param lat_depart (string):  latitude of depart
    :param lon_depart (string):  longitude of depart
    :param city_depart (string): Required. city of depart
    :param address_destination (string): Required. The destination
    :param depart_time (string):  departure hour, in the format '08:00:20'.
    """
    print(address_destination)
    date = "2025-03-29T"
    departure_time = '2024-02-01T' + depart_time
    lat, lon, city = check_city_coordinates(lat_depart,lon_depart,city_depart)
    lat_dest, lon_dest = find_coordinates(address_destination)
    #print(lat_dest, lon_dest)
    
    #print(departure_time)

    r = requests.get('https://api.tomtom.com/routing/1/calculateRoute/{0},{1}:{2},{3}/json?key={4}&departAt={5}'.format(
                        lat_depart,
                        lon_depart,
                        lat_dest,
                        lon_dest,
                        TOMTOM_KEY,
                        departure_time
    ))

    # Parse JSON from the response
    data = r.json()
    #print(data)
    
    #print(data)
    
    result = data['routes'][0]['summary']

    # Calculate distance in kilometers (1 meter = 0.001 kilometers)
    distance_km = result['lengthInMeters'] * 0.001

    # Calculate travel time in minutes (1 second = 1/60 minutes)
    time_minutes = result['travelTimeInSeconds'] / 60
    if time_minutes < 60:
        time_display = f"{time_minutes:.0f} minutes"
    else:
        hours = int(time_minutes / 60)
        minutes = int(time_minutes % 60)
        time_display = f"{hours} hours" + (f" and {minutes} minutes" if minutes > 0 else "")
        
    # Extract arrival time from the JSON structure
    arrival_time_str = result['arrivalTime']

    # Convert string to datetime object
    arrival_time = datetime.fromisoformat(arrival_time_str)

    # Extract and display the arrival hour in HH:MM format
    arrival_hour_display = arrival_time.strftime("%H:%M")


    # return the distance and time
    return(f"The route to go to {address_destination} is {distance_km:.2f} km and {time_display}. Leaving now, the arrival time is estimated at {arrival_hour_display} " )

    
    # Sort the results based on distance
    #sorted_results = sorted(results, key=lambda x: x['dist'])

    #return ". ".join(formatted_results)