import json
import urllib.parse

import requests
from langchain.tools import tool
from loguru import logger

from .common import config, vehicle


# Select coordinates at equal distance, including the last one
def _select_equally_spaced_coordinates(coords, number_of_points=10):
    n = len(coords)
    selected_coords = []
    interval = max((n - 1) / (number_of_points - 1), 1)
    for i in range(number_of_points):
        # Calculate the index, ensuring it doesn't exceed the bounds of the list
        index = int(round(i * interval))
        if index < n:
            selected_coords.append(coords[index])
    return selected_coords


@tool
def search_points_of_interest(search_query: str = "french restaurant"):
    """
    Get some of the closest points of interest matching the query.

    Args:
        search_query (string): Required. Describing the type of point of interest depending on what the user wants to do. Make sure to include the type of POI you are looking for. For example italian restaurant, grocery shop, etc.
    """

    # Extract the latitude and longitude of the vehicle
    vehicle_coordinates = getattr(vehicle, "location_coordinates")
    lat, lon = vehicle_coordinates
    logger.info(f"POI search vehicle's lat: {lat}, lon: {lon}")

    # https://developer.tomtom.com/search-api/documentation/search-service/search
    # Encode the parameters
    # Even with encoding tomtom doesn't return the correct results
    search_query = search_query.replace("'", "")
    encoded_search_query = urllib.parse.quote(search_query)

    # Construct the URL
    url = f"https://api.tomtom.com/search/2/search/{encoded_search_query}.json"
    params = {
        "key": config.TOMTOM_API_KEY,
        "lat": lat,
        "lon": lon,
        "radius": 5000,
        "idxSet": "POI",
        "limit": 50,
    }

    r = requests.get(url, params=params, timeout=5)

    # Parse JSON from the response
    data = r.json()

    logger.debug(f"POI search response: {data}\n url:{url} params: {params}")
    # Extract results
    results = data["results"]

    # TODO: Handle the no results case.
    if not results:
        return "No results found in the vicinity."

    # Sort the results based on distance
    results = sorted(results, key=lambda x: x["dist"])
    # print(sorted_results)

    # Format and limit to top 5 results
    formatted_results = [
        f"{result['poi']['name']}, {int(result['dist'])} meters away"
        for result in results[:3]
    ]

    output = (
        f"There are {len(results)} options in the vicinity. The most relevant are: "
    )
    extra_info = [x["poi"] for x in results[:3]]
    return output + ".\n ".join(formatted_results)


def find_points_of_interest(lat="0", lon="0", type_of_poi="restaurant"):
    """
    Return some of the closest points of interest for a specific location and type of point of interest. The more parameters there are, the more precise.
    :param lat (string):  latitude
    :param lon (string):  longitude
    :param city (string): Required. city
    :param type_of_poi (string): Required. type of point of interest depending on what the user wants to do.
    """
    # https://developer.tomtom.com/search-api/documentation/search-service/points-of-interest-search
    # Encode the parameters
    encoded_type_of_poi = urllib.parse.quote(type_of_poi)

    # Construct the URL
    url = f"https://api.tomtom.com/search/2/search/{encoded_type_of_poi}.json?key={config.TOMTOM_API_KEY}&lat={lat}&lon={lon}&radius=10000&vehicleTypeSet=Car&idxSet=POI&limit=100"

    r = requests.get(url, timeout=5)

    # Parse JSON from the response
    data = r.json()
    # print(data)
    # Extract results
    results = data["results"]

    # Sort the results based on distance
    sorted_results = sorted(results, key=lambda x: x["dist"])
    # print(sorted_results)

    # Format and limit to top 5 results
    formatted_results = [
        f"The {type_of_poi} {result['poi']['name']}, {int(result['dist'])} meters away"
        for result in sorted_results[:5]
    ]

    return ". ".join(formatted_results)


def search_along_route_w_coordinates(points: list[tuple[float, float]], query: str):
    """
    Return some of the closest points of interest along the route/way from the depart point, specified by its coordinates.
    :param points (list[tuple(float, float)]): Required. List of tuples of latitude and longitude of the points along the route.
    :param query (string): Required. type of point of interest depending on what the user wants to do.
    """

    # The API endpoint for searching along a route

    # urlencode the query
    query = urllib.parse.quote(query)

    url = f"https://api.tomtom.com/search/2/searchAlongRoute/{query}.json?key={config.TOMTOM_API_KEY}&maxDetourTime=600&limit=20&sortBy=detourTime"

    points = _select_equally_spaced_coordinates(points, number_of_points=20)

    # The data payload
    payload = {
        "route": {
            "points": [{"lat": pt["latitude"], "lon": pt["longitude"]} for pt in points]
        }
    }

    # Make the POST request
    response = requests.post(url, json=payload, timeout=5)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        # print(json.dumps(data, indent=4))
    else:
        print("Failed to retrieve data:", response.status_code)
        return "Failed to retrieve data. Please try again."
    answer = ""
    if not data["results"]:
        return "No results found along the way."

    if len(data["results"]) == 20:
        answer = "There more than 20 results along the way. Here are the top 3 results:"
    elif len(data["results"]) > 3:
        answer = f"There are {len(data['results'])} results along the way. Here are the top 3 results:"
    for result in data["results"][:3]:
        name = result["poi"]["name"]
        address = result["address"]["freeformAddress"]
        detour_time = result["detourTime"]
        answer = (
            answer
            + f" \n{name} at {address} would require a detour of {int(detour_time/60)} minutes."
        )

    return answer, data["results"][:5]
