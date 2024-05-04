from skills import vehicle


STATUS_TEMPLATE = """We are at {location}, current time: {time}, current date: {date} and our destination is: {destination}.
"""


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
    vs = {
        "location": "Luxembourg Gare, Luxembourg",
        "lat": 49.6000,
        "lon": 6.1333,
        "date": "2025-03-29",
        "time": "08:00:20",
        "destination": "Kirchberg Campus, Kirchberg"
    }
    vs = vehicle.dict()
    return  STATUS_TEMPLATE.format(**vs), vs
