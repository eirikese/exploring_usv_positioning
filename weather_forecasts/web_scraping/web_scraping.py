import requests
from bs4 import BeautifulSoup

def get_weather_from_website(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    # These selectors are placeholders. Replace them with the actual selectors from the website you want to scrape.
    temperature = soup.select_one(".temperature.temperature--warm").text
    # condition = soup.select_one(".condition-class-name").text
    # humidity = soup.select_one(".humidity-class-name").text

    weather_data = {
        "Temperature": temperature,
        # "Condition": condition,
        # "Humidity": humidity
    }

    return weather_data

url = "https://www.yr.no/nb/v%C3%A6rvarsel/daglig-tabell/1-211102/Norge/Tr%C3%B8ndelag/Trondheim/Trondheim"
data = get_weather_from_website(url)
print(data)
