import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import pickle
from ml.model import compute_model_metrics
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
def census_data():
    df = pd.read_csv("./data_clean/census_clean.csv")
    return df

@pytest.fixture
def model():
    model = pickle.load(open("./model/classifier.pkl","rb"))
    return model

""" @pytest.fixture
def model_metrics():
    precision, recall, fbeta = compute_model_metrics(y_slice, feature_preds)
    return precision, recall, fbeta

@pytest.fixture
def processed_data():
    X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    ) """


def test_data(census_data):
    assert not census_data.empty

def test_model(model):
    model_name = type(model).__name__
    assert model_name == "RandomForestClassifier"

