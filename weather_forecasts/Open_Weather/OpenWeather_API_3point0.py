# This weather forecast gathering script uses the OpenWeather Current Weather Data API to get the weather data:
# - https://openweathermap.org/api/one-call-3#current

# Description: Get essential weather data, short-term and long-term forecasts and aggregated weather data is easy with our OpenWeather One Call API 3.0.
# One Call API 3.0 is based on the proprietary OpenWeather Model and is updated every 10 minutes. 
# Thus, in order to receive the most accurate and up-to-date weather data, we recommend you request One Call API 3.0 every 10 minutes.

import requests
from datetime import datetime, timezone

BASE_URL = "https://api.openweathermap.org/data/3.0/onecall?"
API_KEY = "389ba3b6d12aab9f235edb6387763db3"

USER_AGENT = "API_MasterProject orjands@stud.ntnu.no"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse" # To give the location of the coordinates

def get_weather_data(lat, lon, alt):

    url = BASE_URL + "lat=" + lat + "&lon=" + lon + "&alt=" + alt + "&appid=" + API_KEY + "&units=metric"

    print(url)
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return f"Error: {response.status_code} - {response.text}"

def parse_weather_data(data):
    weather_details = {
        # "Location": data.get("name", "Unknown Location"),
        "Last Updated [UTC]": datetime.fromtimestamp(data['current']['dt'], tz=timezone.utc),
        "Air Temperature [℃ ]": data["current"]["temp"],
        "Relative Humidity [%]": data["current"]["humidity"],
        "Wind Speed [m/s]": data["current"]["wind_speed"],
        "Gust Speed [m/s]": data["current"]["wind_gust"],
        "Wind Direction [deg]": data["current"]["wind_deg"],
        "Weather Condition": data["current"]["weather"][0]["main"],
        "Weather Description": data["current"]["weather"][0]["description"],
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