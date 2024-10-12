import requests

response = requests.post('cg-census-pred-app.herokuapp.com')

print(response.status_code)
print(response.json())