import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from ml.model import compute_model_metrics, inference
from ml.data import process_data

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

@pytest.fixture
def model_outputs():
    model = pickle.load(open("./model/classifier.pkl","rb"))
    encoder = pickle.load(open("./model/encoder.pkl","rb"))
    lb = pickle.load(open("./model/lb.pkl","rb"))
    return model, encoder, lb

@pytest.fixture
def census_data_raw():
    df = pd.read_csv("./data_clean/census_clean.csv")
    return df

@pytest.fixture
def data_split(census_data_raw):
    data = census_data_raw
    train, test = train_test_split(data, test_size=0.20)
    return train, test

@pytest.fixture
def processed_data(data_split, model_outputs):
    _, test = data_split
    _, encoder, lb = model_outputs
    X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    return X_test, y_test

@pytest.fixture
def predictions(processed_data, model_outputs):
    X_test, _ = processed_data
    model, _, _ = model_outputs
    preds = inference(model, X_test)
    return preds

@pytest.fixture
def model_metrics(processed_data, predictions):
    _ , y_test = processed_data
    preds = predictions
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    return precision, recall, fbeta

def test_data(census_data_raw):
    assert not census_data_raw.empty

def test_model(model_outputs):
    model, _, _ = model_outputs
    model_name = type(model).__name__
    assert model_name == "RandomForestClassifier"

def test_model_metrics(model_metrics):
    _, _, fbeta = model_metrics
    assert fbeta >= 0.5

def test_inference(predictions):
    preds = predictions
    assert len(preds.tolist()) > 0

