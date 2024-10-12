import requests

test_data = {
        'age': 50,
        'workclass': 'Private',
        'fnlgt': 280464,
        'education': 'Some-college',
        'education-num': 10,
        'marital-status': 'Married',
        'occupation': 'Exec-managerial',
        'relationship': 'Husband',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 60,
        'native-country': 'United-States'
    }

response_get = requests.get('https://cg-census-pred-app-prod-e1af23d4a843.herokuapp.com/')
print(response_get.json())

response = requests.post('https://cg-census-pred-app-prod-e1af23d4a843.herokuapp.com/prediction', json=test_data) 
assert response.status_code == 200

print(response.status_code)
print(response.json())