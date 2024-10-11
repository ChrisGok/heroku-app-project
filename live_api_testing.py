import requests

response = requests.post('url/to/query')

print(response.status_code)
print(response.json())