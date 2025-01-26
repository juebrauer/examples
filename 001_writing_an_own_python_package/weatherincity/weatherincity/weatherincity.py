"""
Weather module

Currently only provides one function in order to
retrieve the temperature.
"""

import sys
import requests

def get_current_temperature(city: str) -> str:
    """
    Returns the current temperatur using the website wttr.in

    Note: The temperatur and the unit is returned as a string
    """

    url = f"https://wttr.in/{city}?format=%t"
    try:
        response = requests.get(url)

        # status code 200 means OK
        if response.status_code == 200:
            return response.text.strip()
        else:
            return f"Sorry! Error: Server response is {response.status_code}"
        
    except requests.exceptions.RequestException as e:
        return f"Request error: {e}"


# Is this file used as a script (__main__) or as module?
if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("Call me like this: python weather.py Berlin")
        sys.exit(1)
    
    city = sys.argv[1]
    temperature = get_current_temperature(city)
    print(f"The current temperature in {city} is: {temperature}")
