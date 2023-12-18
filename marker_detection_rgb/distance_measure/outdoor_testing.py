import requests, json
from datetime import datetime, timezone

USER_AGENT = "API_MasterProject orjands@stud.ntnu.no"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse" # To give the location of the coordinates

LATITUDE = "63.416428"
LONGTITUDE = "10.410296"
ALTITUDE = "0.0"

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


#########################################################################
# WEATHER FORECAST SCRIPT: OpenWeatherMap

BASE_URL_OPEN = "http://api.openweathermap.org/data/2.5/weather?"
API_KEY = "389ba3b6d12aab9f235edb6387763db3"

def get_weather_data(lat, lon, alt):

    url = BASE_URL_OPEN + "lat=" + lat + "&lon=" + lon + "&alt=" + alt + "&appid=" + API_KEY + "&units=metric"

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


#########################################################################
# WEATHER FORECAST SCRIPT: Metrologisk Institutt Norge

BASE_URL_MET = "https://api.met.no/weatherapi/"

# This is the one being printed to the Terinal
def get_nowcast_data(lat, lon):
    # This updates every 5 minutes!
    # Construct the URL for the Nowcast API
    endpoint = f"nowcast/2.0/complete?lat={lat}&lon={lon}"
    
    
    url = BASE_URL_MET + endpoint
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

def round_coordinates(lat, lon, alt, precision=4):
    return round(float(lat), precision), round(float(lon), precision), round(float(alt), precision)


#########################################################################
# VIDEO CAPTURE AND MARKER DETECTION

import cv2
from cv2 import aruco
import numpy as np
import csv

# Load in the calibration data
calib_data_path = "../calibration_data/MultiMatrix.npz"
calib_data = np.load(calib_data_path)

cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]

# Marker size in centimeters for distance measurement
MARKER_SIZE_SMALL = 26.28
MARKER_SIZE_BIG = 41.2

# List of dictionaries to use for detection
dict_list = [
    aruco.DICT_4X4_50,
    aruco.DICT_5X5_50,
    aruco.DICT_6X6_50,
    aruco.DICT_7X7_50
]

param_markers = aruco.DetectorParameters()
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Prepare CSV file
csv_file = open('distance_measurements_8m_1020.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Iteration', 'Number of markers detected', 'Mean Distance (cm)', 'Mean Distance (m)'])

iteration_count = 0
max_iterations = 500  # Set the maximum number of iterations

while iteration_count < max_iterations:
    ret, frame = cap.read()
    if not ret:
        break

    iteration_count += 1
    total_distance_cm = 0
    total_distance_m = 0
    marker_count = 0

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Iterate over the list of dictionaries for detection
    for dictionary in dict_list:
        marker_dict = aruco.getPredefinedDictionary(dictionary)
        marker_corners, marker_IDs, _ = aruco.detectMarkers(gray_frame, marker_dict, parameters=param_markers)

        if marker_corners:
            rVec, tVec, _ = aruco.estimatePoseSingleMarkers(marker_corners, MARKER_SIZE_SMALL, cam_mat, dist_coef)
            total_markers = range(0, marker_IDs.size)

            for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
                cv2.polylines(
                    frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv2.LINE_AA
                )
                corners = corners.reshape(4, 2)
                corners = corners.astype(int)
                top_left = corners[0].ravel()
                top_right = corners[1].ravel()
                bottom_right = corners[2].ravel()
                bottom_left = corners[3].ravel()

                # Calculating the distance
                distance_cm = np.sqrt(
                    tVec[i][0][2] ** 2 + tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2
                )
                distance_m = distance_cm * 0.01
                
                total_distance_cm += distance_cm
                total_distance_m += distance_m
                marker_count += 1
                
                # Draw the pose of the marker
                point = cv2.drawFrameAxes(frame, cam_mat, dist_coef, rVec[i], tVec[i], 4, 4)
                
                
    # Compute the mean distance
    if marker_count > 0:
        mean_distance_cm = total_distance_cm / marker_count
        mean_distance_m = total_distance_m / marker_count
        # Write to CSV
        csv_writer.writerow([iteration_count, marker_count, round(mean_distance_cm, 2), round(mean_distance_m, 4)])

        # Display mean distance in the frame
        cv2.putText(frame, f"Mean Dist [cm]: {round(mean_distance_cm, 2)}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 255, 0), 2)
        cv2.putText(frame, f"Mean Dist [m]: {round(mean_distance_m, 4)}", (10, 60), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 255, 0), 2)

    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)

    if key == ord("q"):  # Allow early exit
        break
    

place_name = get_place_name(LATITUDE, LONGTITUDE)
print(f"Weather Data for: {place_name}\n")


weather_data_OPEN = get_weather_data(LATITUDE, LONGTITUDE, ALTITUDE)
print("OpenWeather DATA:")
if isinstance(weather_data_OPEN, dict):  # Check if the response is a dictionary
    current_weather_data = parse_weather_data(weather_data_OPEN)
    for key, value in current_weather_data.items():
        print(f"{key}: {value}")
else:
    print(weather_data_OPEN)  # Print the error message


round_coordinates(LATITUDE, LONGTITUDE, ALTITUDE)
weather_data_MET = get_nowcast_data(LATITUDE, LONGTITUDE)
print("\nMET DATA:")
for key, value in weather_data_MET.items():
    print(f"{key}: {value}")
       
cap.release()
cv2.destroyAllWindows()
csv_file.close()