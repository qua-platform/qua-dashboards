import requests
from qua_dashboards.utils.data_serialisation import serialise_data


def send_data_to_dash(data, url="http://localhost:8050"):
    serialised_data = serialise_data(data)
    response = requests.post(f"{url}/update-data", json=serialised_data)
    if response.ok:
        print("Data sent successfully")
        return True
    else:
        print("Failed to send data")
        return False
