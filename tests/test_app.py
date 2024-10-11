from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

def test_get():
    response = client.get("/")
    print(response.json())
    assert response.status_code == 200
    assert response.json() == 'Heyho! Nice to see you!'

def test_post_data_success():
    input_data = {
                'age': 35,
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
    response = client.post('/prediction', json=input_data)
    assert response.status_code == 200
    assert response.json() == {'prediction': '>50k'}


def test_post_data_fail():
    input_data = {
                'age': 55,
                'workclass': 'Self-emp-not-inc',
                'fnlgt': 292175,
                'education': '7th-8th',
                'education-num': 1,
                'marital-status': 'Divorced',
                'occupation': 'Transport-moving',
                'relationship': 'Unmarried',
                'race': 'White',
                'sex': 'Male',
                'capital-gain': 0,
                'capital-loss': 0,
                'hours-per-week': 45,
                'native-country': 'Amer-Indian-Eskimo'
        }
    response = client.post('/prediction', json=input_data)
    assert response.status_code == 200
    assert response.json() == {'prediction': '<=50k'}