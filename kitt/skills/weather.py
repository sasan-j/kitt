import requests
from loguru import logger
from langchain.tools import tool

from .common import config, vehicle


def get_weather_current_location():
    """
    Returns the CURRENT weather in current location.
    When responding to user, only mention the weather condition, temperature, and the temperature that it feels like, unless the user asks for more information.

    Returns:
        dict: The weather data in the specified location.
    """
    print(
        f"get_weather: location is empty, using the vehicle location. ({vehicle.location})"
    )
    location = vehicle.location
    return get_weather(location)


@tool
def get_weather(location: str = "here"):
    """
    Get the current weather in a specified location.
    When responding to user, only mention the weather condition, temperature, and the temperature that it feels like, unless the user asks for more information.

    Args:
        location (string) : Optional. The name of the location, if empty or here, the vehicle location is used.

    Returns:
        dict: The weather data in the specified location.
    """

    if location == "" or location == "here":
        logger.warning(
            f"get_weather: location is empty, using the vehicle location. ({vehicle.location})"
        )
        location = vehicle.location

    # The endpoint URL provided by WeatherAPI
    url = f"http://api.weatherapi.com/v1/current.json?key={config.WEATHER_API_KEY}&q={location}&aqi=no"

    # Make the API request
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Failed to get weather data: {response.status_code}, {response.text}")
        return f"Failed to get weather data, try again", response

    # Parse the JSON response
    weather_data = response.json()

    # Extracting the necessary pieces of data
    location = weather_data["location"]["name"]
    region = weather_data["location"]["region"]
    country = weather_data["location"]["country"]
    time = weather_data["location"]["localtime"]
    temperature_c = weather_data["current"]["temp_c"]
    condition_text = weather_data["current"]["condition"]["text"]
    if "wind_kph" in weather_data["current"]:
        wind_kph = weather_data["current"]["wind_kph"]
    humidity = weather_data["current"]["humidity"]
    feelslike_c = weather_data["current"]["feelslike_c"]

    # Formulate the sentences - {region}, {country}
    weather_sentences = (
        f"The current weather in {location} is {condition_text} "
        f"with a temperature of {temperature_c}°C that feels like {feelslike_c}°C. "
        # f"Humidity is at {humidity}%. "
        # f"Wind speed is {wind_kph} kph." if 'wind_kph' in weather_data['current'] else ""
    )
    # return weather_sentences, weather_data
    return weather_sentences


# weather forecast API
def get_forecast(city_name: str = "", when=0, **kwargs):
    """
    Get the weather forecast in a specified number of days for a specified location.
    Args:
        city_name (string) : Required. The name of the city.
        when (int) : Required. in number of days (until the day for which we want to know the forecast) (example: tomorrow is 1, in two days is 2, etc.)
    """

    when += 1
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
        location = data.get("location", {})
        city_name = location.get("name", "the specified location")

        # print(data)

        # Extract the forecast days
        forecast_days = data.get("forecast", {}).get("forecastday", [])[when - 1 :]
        # number = 0

        # print (forecast_days)

        for day in forecast_days:
            date = day.get("date", "a specific day")
            conditions = (
                day.get("day", {})
                .get("condition", {})
                .get("text", "weather conditions")
            )
            max_temp_c = day.get("day", {}).get("maxtemp_c", "N/A")
            min_temp_c = day.get("day", {}).get("mintemp_c", "N/A")
            chance_of_rain = day.get("day", {}).get("daily_chance_of_rain", "N/A")

            if when == 1:
                number_str = "today"
            elif when == 2:
                number_str = "tomorrow"
            else:
                number_str = f"in {when-1} days"

            # Generate a sentence for the day's forecast
            forecast_sentence = f"On {date} ({number_str}) in {city_name}, the weather will be {conditions} with a high of {max_temp_c}C and a low of {min_temp_c}C. There's a {chance_of_rain}% chance of rain. "

            # number = number + 1
            # Add the sentence to the result
            forecast_sentences += forecast_sentence
        return forecast_sentences
    else:
        # Handle errors
        print(f"Failed to get weather data: {response.status_code}, {response.text}")
        return f"error {response.status_code}"
