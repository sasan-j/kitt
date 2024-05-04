from datetime import datetime

import requests

from .common import config, vehicle


def find_coordinates(address):
    """
    Find the coordinates of a specific address.
    :param address (string): Required. The address
    """
    url = f"https://api.tomtom.com/search/2/geocode/{address}.json?key={config.TOMTOM_API_KEY}"
    response = requests.get(url)
    data = response.json()
    lat = data["results"][0]["position"]["lat"]
    lon = data["results"][0]["position"]["lon"]
    return lat, lon


def calculate_route():
    api_key = "api_key"
    origin = "49.631997,6.171029"
    destination = "49.586745,6.140002"

    url = f"https://api.tomtom.com/routing/1/calculateRoute/{origin}:{destination}/json?key={api_key}"
    response = requests.get(url)
    data = response.json()

    lats = []
    lons = []

    for point in data["routes"][0]["legs"][0]["points"]:
        lats.append(point["latitude"])
        lons.append(point["longitude"])
    # fig = px.line_geo(lat=lats, lon=lons)
    # fig.update_geos(fitbounds="locations")

    fig = px.line_mapbox(lat=lats, lon=lons, zoom=12, height=600)

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=12,
        mapbox_center_lat=lats[0],
        mapbox_center_lon=lons[0],
    )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    return fig


def find_route_tomtom(
    lat_depart="0",
    lon_depart="0",
    lat_dest="0",
    lon_dest="0",
    depart_datetime="",
    **kwargs,
):
    """
    Return the distance and the estimated time to go to a specific destination from the current place, at a specified depart time.
    :param lat_depart (string):  latitude of depart
    :param lon_depart (string):  longitude of depart
    :param lat_dest (string):  latitude of destination
    :param lon_dest (string):  longitude of destination
    :param depart_time (string):  departure hour, in the format '08:00:20'.
    """

    r = requests.get(
        f"https://api.tomtom.com/routing/1/calculateRoute/{lat_depart},{lon_depart}:{lat_dest},{lon_dest}/json?key={config.TOMTOM_API_KEY}&departAt={depart_datetime}",
        timeout=5,
    )

    # Parse JSON from the response
    response = r.json()

    result = response["routes"][0]["summary"]



    distance_m = result["lengthInMeters"]
    duration_s = result["travelTimeInSeconds"]
    arrival_time = result["arrivalTime"]
    # Convert string to datetime object
    arrival_time = datetime.fromisoformat(arrival_time)

    return {
        "distance_m": distance_m,
        "duration_s": duration_s,
        "arrival_time": arrival_time,
    }, response


def find_route(destination=""):
    """
    Find a route and return the distance and the estimated time to go to a specific destination from the current location.
    :param destination (string): Required. The destination
    """
    # lat, lon, city = check_city_coordinates(lat_depart,lon_depart,city_depart)
    lat_dest, lon_dest = find_coordinates(destination)
    print(f"lat_dest: {lat_dest}, lon_dest: {lon_dest}")

    # Extract the latitude and longitude of the vehicle
    vehicle_coordinates = getattr(vehicle, "location_coordinates")
    lat_depart, lon_depart = vehicle_coordinates
    print(f"lat_depart: {lat_depart}, lon_depart: {lon_depart}")

    date = getattr(vehicle, "date")
    time = getattr(vehicle, "time")
    departure_time = f"{date}T{time}"

    trip_info, raw_response = find_route_tomtom(
        lat_depart, lon_depart, lat_dest, lon_dest, departure_time
    )

    distance, duration, arrival_time = trip_info["distance_m"], trip_info["duration_s"], trip_info["arrival_time"]

    # Calculate distance in kilometers (1 meter = 0.001 kilometers)
    distance_km = distance * 0.001
    # Calculate travel time in minutes (1 second = 1/60 minutes)
    time_minutes = duration / 60
    if time_minutes < 60:
        time_display = f"{time_minutes:.0f} minutes"
    else:
        hours = int(time_minutes / 60)
        minutes = int(time_minutes % 60)
        time_display = f"{hours} hours" + (
            f" and {minutes} minutes" if minutes > 0 else ""
        )

    # Extract and display the arrival hour in HH:MM format
    arrival_hour_display = arrival_time.strftime("%H:%M")

    # return the distance and time
    return f"This is the answer you must copy exactly as is: The route to {destination} is {distance_km:.2f} km and {time_display}. Leaving now, the arrival time is estimated at {arrival_hour_display} "
