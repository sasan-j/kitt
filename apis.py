import requests

from geopy.geocoders import Nominatim


###################################################
# Functions we want to articulate (APIs calls): ###
###################################################

########################################################################################
# Functions called  in the articulated functions (not directly called by the model): ###
########################################################################################

geolocator = Nominatim(user_agent="MyApp")

def find_precise_place(lat, lon):
    location = geolocator.reverse(str(lat) +", " + str(lon))
    return location.raw.get('display_name', {})

def find_coordinates(address):
    coord = geolocator.geocode(address)
    lat = coord.latitude
    lon = coord.longitude
    return(lat,lon)


def check_city_coordinates(lat = "", lon = "", city = "", **kwargs):
    """
    :param lat: latitude
    :param lon: longitude
    :param city: name of the city

    Checks if the coordinates correspond to the city, if not update the coordinate to correspond to the city
    """
    if lat != "0" and lon != "0":
        reverse = partial(geolocator.reverse, language="en")
        location = reverse(f"{lat}, {lon}")
        address = location.raw.get('address', {})
        city = address.get('city') or address.get('town') or address.get('village') or address.get('county')
    else : 
        reverse = partial(geolocator.reverse, language="en")
        location = reverse(f"{lat}, {lon}")
        address = location.raw.get('address', {})
        city_name = address.get('city') or address.get('town') or address.get('village') or address.get('county')
        if city_name is None :
            city_name = 'not_found'
        print(city_name)
        if city_name.lower() != city.lower():
            coord = geolocator.geocode(city )
            if coord is None:
                coord = geolocator.geocode(city)
            lat = coord.latitude
            lon = coord.longitude
    return lat, lon, city

# Select coordinates at equal distance, including the last one
def select_equally_spaced_coordinates(coords, number_of_points=10):
    n = len(coords)
    selected_coords = []
    interval = max((n - 1) / (number_of_points - 1), 1)
    for i in range(number_of_points):
        # Calculate the index, ensuring it doesn't exceed the bounds of the list
        index = int(round(i * interval))
        if index < n:
            selected_coords.append(coords[index])
    return selected_coords

def find_points_of_interest(lat="0", lon="0", city="", type_of_poi="restaurant", **kwargs):
    """
    Return some of the closest points of interest for a specific location and type of point of interest. The more parameters there are, the more precise.
    :param lat (string):  latitude
    :param lon (string):  longitude
    :param city (string): Required. city
    :param type_of_poi (string): Required. type of point of interest depending on what the user wants to do.
    """
    lat, lon, city = check_city_coordinates(lat,lon,city)

    r = requests.get(f'https://api.tomtom.com/search/2/search/{type_of_poi}'
                     '.json?key={0}&lat={1}&lon={2}&radius=10000&idxSet=POI&limit=100'.format(
                        TOMTOM_KEY,
                        lat,
                        lon
    ))

    # Parse JSON from the response
    data = r.json()
    #print(data)
    # Extract results
    results = data['results']

    # Sort the results based on distance
    sorted_results = sorted(results, key=lambda x: x['dist'])
    #print(sorted_results)

    # Format and limit to top 5 results
    formatted_results = [
        f"The {type_of_poi} {result['poi']['name']} is {int(result['dist'])} meters away"
        for result in sorted_results[:5]
    ]


    return ". ".join(formatted_results)

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


def search_along_route(latitude_depart, longitude_depart, city_destination, type_of_poi):
    """
    Return some of the closest points of interest along the route from the depart point, specified by its coordinates and a city destination.
    :param latitude_depart (string):  Required. Latitude of depart location
    :param longitude_depart (string):  Required. Longitude of depart location
    :param city_destination (string): Required. City destination
    :param type_of_poi (string): Required. type of point of interest depending on what the user wants to do.
    """
    
    lat_dest, lon_dest = find_coordinates(city_destination)
    print(lat_dest)
    
    r = requests.get('https://api.tomtom.com/routing/1/calculateRoute/{0},{1}:{2},{3}/json?key={4}'.format(
                        latitude_depart,
                        longitude_depart,
                        lat_dest,
                        lon_dest,
                        TOMTOM_KEY
    ))
    
    coord_route = select_equally_spaced_coordinates(r.json()['routes'][0]['legs'][0]['points'])

    # The API endpoint for searching along a route
    url = f'https://api.tomtom.com/search/2/searchAlongRoute/{type_of_poi}.json?key={TOMTOM_KEY}&maxDetourTime=700&limit=20&sortBy=detourTime'

    # The data payload
    payload = {
      "route": {
        "points": [
          {"lat": float(latitude_depart), "lon": float(longitude_depart)},
          {"lat": float(coord_route[1]['latitude']), "lon": float(coord_route[1]['longitude'])},
          {"lat": float(coord_route[2]['latitude']), "lon": float(coord_route[2]['longitude'])},
          {"lat": float(coord_route[3]['latitude']), "lon": float(coord_route[3]['longitude'])},
          {"lat": float(coord_route[4]['latitude']), "lon": float(coord_route[4]['longitude'])},
          {"lat": float(coord_route[5]['latitude']), "lon": float(coord_route[5]['longitude'])},
          {"lat": float(coord_route[6]['latitude']), "lon": float(coord_route[6]['longitude'])},
          {"lat": float(coord_route[7]['latitude']), "lon": float(coord_route[7]['longitude'])},
          {"lat": float(coord_route[8]['latitude']), "lon": float(coord_route[8]['longitude'])},
          {"lat": float(lat_dest), "lon": float(lon_dest)},
        ]
      }
    }

    # Make the POST request
    response = requests.post(url, json=payload)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        print(json.dumps(data, indent=4))
    else:
        print('Failed to retrieve data:', response.status_code)
    answer = ""
    for result in data['results']:
        name = result['poi']['name']
        address = result['address']['freeformAddress']
        detour_time = result['detourTime']
        answer = answer + f" \nAlong the route to {city_destination}, there is the {name} at {address} that would represent a detour of {int(detour_time/60)} minutes."
        
    return answer


#current weather API
def get_weather(city_name:str= "", **kwargs):
    """
    Returns the CURRENT weather in a specified city.
    Args:
    city_name (string) : Required. The name of the city.
    """
    # The endpoint URL provided by WeatherAPI
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city_name}&aqi=no"

    # Make the API request
    response = requests.get(url)

    if response.status_code == 200:
        # Parse the JSON response
        weather_data = response.json()

        # Extracting the necessary pieces of data
        location = weather_data['location']['name']
        region = weather_data['location']['region']
        country = weather_data['location']['country']
        time = weather_data['location']['localtime']
        temperature_c = weather_data['current']['temp_c']
        condition_text = weather_data['current']['condition']['text']
        wind_mph = weather_data['current']['wind_mph']
        humidity = weather_data['current']['humidity']
        feelslike_c = weather_data['current']['feelslike_c']

        # Formulate the sentences
        weather_sentences = (
            f"The current weather in {location}, {region}, {country} is {condition_text} "
            f"with a temperature of {temperature_c}°C that feels like {feelslike_c}°C. "
            f"Humidity is at {humidity}%. "
            f"Wind speed is {wind_mph} mph."
        )
        return weather_sentences
    else:
        # Handle errors
        return f"Failed to get weather data: {response.status_code}, {response.text}"



#weather forecast API
def get_forecast(city_name:str= "", when = 0, **kwargs):
    """
    Returns the weather forecast in a specified number of days for a specified city .
    Args:
    city_name (string) : Required. The name of the city.
    when (int) : Required. in number of days (until the day for which we want to know the forecast) (example: tomorrow is 1, in two days is 2, etc.)
    """
    #print(when)
    when +=1
    # The endpoint URL provided by WeatherAPI
    url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q={city_name}&days={str(when)}&aqi=no"


    # Make the API request
    response = requests.get(url)

    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        
        # Initialize an empty string to hold our result
        forecast_sentences = ""

        # Extract city information
        location = data.get('location', {})
        city_name = location.get('name', 'the specified location')
        
        #print(data)
    

        # Extract the forecast days
        forecast_days = data.get('forecast', {}).get('forecastday', [])[when-1:]
        #number = 0
        
        #print (forecast_days)

        for day in forecast_days:
            date = day.get('date', 'a specific day')
            conditions = day.get('day', {}).get('condition', {}).get('text', 'weather conditions')
            max_temp_c = day.get('day', {}).get('maxtemp_c', 'N/A')
            min_temp_c = day.get('day', {}).get('mintemp_c', 'N/A')
            chance_of_rain = day.get('day', {}).get('daily_chance_of_rain', 'N/A')
            
            if when == 1:
                number_str = 'today'
            elif when == 2:
                number_str = 'tomorrow'
            else:
                number_str = f'in {when-1} days'

            # Generate a sentence for the day's forecast
            forecast_sentence = f"On {date} ({number_str}) in {city_name}, the weather will be {conditions} with a high of {max_temp_c}°C and a low of {min_temp_c}°C. There's a {chance_of_rain}% chance of rain. "
            
            #number = number + 1
            # Add the sentence to the result
            forecast_sentences += forecast_sentence
        return forecast_sentences
    else:
        # Handle errors
        print( f"Failed to get weather data: {response.status_code}, {response.text}")
        return f'error {response.status_code}'
    

