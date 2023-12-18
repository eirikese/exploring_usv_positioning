# This weather forecast gathering script uses the OpenWeather Current Weather Data API to get the weather data:
# - https://openweathermap.org/current

# Description: Access current weather data for any location on Earth! 
# We collect and process weather data fromdifferent sources such as global and local weather models, satellites, radars and a vast network of weather stations. 
# Data is available in JSON, XML, or HTML format 
# Forecasts are updated every change in weather. It is our best estimate of the current weather conditions. 

import requests
from datetime import datetime, timezone

BASE_URL = "http://api.openweathermap.org/data/2.5/weather?"
API_KEY = "389ba3b6d12aab9f235edb6387763db3"

USER_AGENT = "API_MasterProject orjands@stud.ntnu.no"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse" # To give the location of the coordinates

def get_weather_data(lat, lon, alt):

    url = BASE_URL + "lat=" + lat + "&lon=" + lon + "&alt=" + alt + "&appid=" + API_KEY + "&units=metric"

    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return f"Error: {response.status_code} - {response.text}"

def parse_weather_data(data):
    weather_details = {
        # "Location": data.get("name", "Unknown Location"),
        "Last Updated [UTC]": datetime.fromtimestamp(data['dt'], tz=timezone.utc),
        "Air Temperature [℃ ]": data["main"]["temp"],
        "Relative Humidity [%]": data["main"]["humidity"],
        "Wind Speed [m/s]": data["wind"]["speed"],
        "Gust Speed [m/s]": data["wind"]["gust"],
        "Wind Direction [deg]": data["wind"]["deg"],
        "Weather Condition": data["weather"][0]["main"],
        "Weather Description": data["weather"][0]["description"],
        # "Cloudiness [%]": data["clouds"]["all"],
        # "Pressure [hPa]": data["main"]["pressure"]
    }
    return weather_details

def get_place_name(lat, lon):
    params = {
        "lat": lat,
        "lon": lon,
        "format": "json"
    }

    headers = {
        "User-Agent": USER_AGENT
    }

    response = requests.get(NOMINATIM_URL, params=params, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        return data.get("display_name", "Unknown Location")
    else:
        return "Unknown Location"

if __name__ == "__main__":
    lat = input("Enter latitude coordinate: ")
    lon = input("Enter longitude coordinate: ")
    alt = input("Enter altitude [m]: ")

    weather_data = get_weather_data(lat, lon, alt)
    
    place_name = get_place_name(lat, lon)
    print(f"Weather Data for: {place_name}\n")
    
    if isinstance(weather_data, dict):  # Check if the response is a dictionary
        current_weather_data = parse_weather_data(weather_data)
        for key, value in current_weather_data.items():
            print(f"{key}: {value}")
    else:
        print(weather_data)  # Print the error message