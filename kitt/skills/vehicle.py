from .common import vehicle, Speed


STATUS_TEMPLATE = """The current location is: {location} 
Current coordinates: {lat}, {lon}
The current date and time: {date} {time}
The current destination is: {destination}"""


def vehicle_status() -> tuple[str, dict[str, str]]:
    """Get current vehicle status, which includes, location, date, time, destination.
    Call this to get the current destination or location of the car/vehicle.
    Returns:
        dict[str, str]: The vehicle status. For example:
        {
            "location": "Luxembourg Gare, Luxembourg",
            "lat": 49.6000,
            "lon": 6.1333,
            "date": "2025-03-29",
            "time": "08:00:20",
            "destination": "Kirchberg Campus, Kirchberg"
        }
    """
    # vs = {
    #     "location": "Luxembourg Gare, Luxembourg",
    #     "lat": 49.6000,
    #     "lon": 6.1333,
    #     "date": "2025-03-29",
    #     "time": "08:00:20",
    #     "destination": "Kirchberg Campus, Luxembourg"
    # }
    vs = vehicle.dict()
    vs["lat"] = vs["location_coordinates"][0]
    vs["lon"] = vs["location_coordinates"][1]
    return  STATUS_TEMPLATE.format(**vs), vs



def set_vehicle_speed(speed: Speed):
    """Set the speed of the vehicle.
    Args:
        speed (Speed): The speed of the vehicle. ("slow", "fast")
    """
    vehicle.speed = speed
    return f"The vehicle speed is set to {speed.value}."

def set_vehicle_destination(destination: str):
    """Set the destination of the vehicle.
    Args:
        destination (str): The destination of the vehicle.
    """
    vehicle.destination = destination
    return f"The vehicle destination is set to {destination}."
