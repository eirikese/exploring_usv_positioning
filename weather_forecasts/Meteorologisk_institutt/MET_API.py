# This weather forecast gathering script uses the Meteorologisk institutt (MET) nowcast 2.0 API to get the weather data:
# - https://api.met.no/weatherapi/nowcast/2.0/documentation

# Description: This service delivers two-hour weather forecasts in JSON or XML for locations in Norway, Sweden, Finland and Denmark. 
# Forecasts are updated every 5 minutes. It is our best estimate of the current weather conditions.

import requests, json

BASE_URL = "https://api.met.no/weatherapi/"
USER_AGENT = # INSERT USER AGENT

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse" 

def get_weather_data(lat, lon):
    # Construct the URL
    endpoint = f"locationforecast/2.0/compact?lat={lat}&lon={lon}"
    url = BASE_URL + endpoint

    # Define the headers
    headers = {
        "User-Agent": USER_AGENT
    }

    # Send the GET request
    response = requests.get(url, headers=headers)

    # Check the response
    if response.status_code == 200:
        data = response.json()
        # Extract the newest weather data
        newest_data = data['properties']['timeseries'] # [2]
        
        formatted_data = json.dumps(response.json(), indent=4)
        formatted_newest_data = json.dumps(newest_data, indent=4)

        return formatted_newest_data
    else:
        return f"Error: {response.status_code} - {response.text}"

# This is the one being printed to the Terinal
def get_nowcast_data(lat, lon, altitude=None):
    
    # Construct the URL for the Nowcast API
    endpoint = f"nowcast/2.0/complete?lat={lat}&lon={lon}"
    
    if altitude is not None:
        endpoint += f"&altitude={altitude}"
    else:
        altitude = 0.0
        endpoint += f"&altitude={altitude}"
    
    url = BASE_URL + endpoint
    headers = {
        "User-Agent": USER_AGENT
    }
    
    # Send the GET request
    response = requests.get(url, headers=headers)

    # Check the response
    if response.status_code == 200:
        data = response.json()
        radar_coverage = data['properties']['meta']['radar_coverage']
        
        if radar_coverage == "ok":
            # Extract the most current forecast data
            current_data = data['properties']['timeseries'][0]
            
            # Process the data as needed
            time = current_data['time']
            air_temperature = current_data['data']['instant']['details']['air_temperature']
            precipitation_rate = current_data['data']['instant']['details']['precipitation_rate']
            precipitation_amount_next_hour = current_data['data']['next_1_hours']['details']['precipitation_amount']
            relative_humidity = current_data['data']['instant']['details']['relative_humidity']
            wind_from_direction = current_data['data']['instant']['details']['wind_from_direction']
            wind_speed = current_data['data']['instant']['details']['wind_speed']
            wind_speed_of_gust = current_data['data']['instant']['details']['wind_speed_of_gust']
            weather_summary = current_data['data']['next_1_hours']['summary']['symbol_code']
            last_updated = data['properties']['meta']['updated_at']
             
            weather_details = {
                "Last Updated [UTC]": last_updated,
                # "Time": time,
                "Air Temperature [℃ ]": air_temperature,
                # "Precipitation_rate [mm/h]": precipitation_rate,
                "Relative Humidity [%]": relative_humidity,
                "Wind Speed [m/s]": wind_speed,
                "Gust Speed [m/s]": wind_speed_of_gust,
                "Wind From Direction [deg]": wind_from_direction,
                "Weather Summary next hour": weather_summary,
                "Precipitation next hour [mm]": precipitation_amount_next_hour
            }
        else:
            weather_details = f"No accurate weather data available due to radar coverage status: {radar_coverage}"
        return weather_details
    else:
        return f"Error: {response.status_code} - {response.text}"

def round_coordinates(lat, lon, precision=4):
    return round(float(lat), precision), round(float(lon), precision)

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
    # Example usage
    lat = input("Enter latitude coordinate: ")
    lon = input("Enter longitude coordinate: ")
    alt = input("Enter altitude [m]: ")
    
    place_name = get_place_name(lat, lon)
    print(f"Weather Data for: {place_name}\n")
    
    round_coordinates(lat, lon)

    weather_data = get_weather_data(lat, lon)
    # print(weather_data)
    
    current_weather_data = get_nowcast_data(lat, lon, alt)
    for key, value in current_weather_data.items():
        print(f"{key}: {value}")
    

    """
    Finally, using six (6) decimal places, you get to the exact point! 
    So, when it comes to GPS data, you need coordinates out to at least five (5) decimal places to be 
    useable in terms of locating and subsequently evaluating incremental speeds between points.
    
    This API can only use coordinates with 4 decimals!
    - https://www.forensicdjs.com/gps-coordinates-many-decimal-places-need/ (Examples of how accurate 4 decimal coordinates are)
    """