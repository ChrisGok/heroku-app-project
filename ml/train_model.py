# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import os
import json
import pandas as pd
import pickle
from data import process_data
from model import train_model, compute_model_metrics, inference

# Add code to load in the data.

data = pd.read_csv(os.path.join('../','data_clean','census_clean.csv'))

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.

X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.

model = train_model(X_train, y_train)

pickle.dump(model, open('../model/classifier.pkl', 'wb'))
pickle.dump(encoder, open('../model/encoder.pkl', 'wb'))
pickle.dump(lb, open('../model/lb.pkl', 'wb'))


def slice_data(df, cat_feature):
    """
    Function which provides ml model metrics on data slices
    """

    # initialize object to store results
    results = {}

    for feature in df[cat_feature].unique():

        df_temp = df[df[cat_feature] == feature]

        # process data to calculate metrics as for the full model
        X_slice, y_slice, _, _ = process_data(
            df_temp, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
        )

        # make predictions 
        feature_preds = inference(model=model, X=X_slice)

        # calculate metrics
        precision, recall, fbeta = compute_model_metrics(y_slice, feature_preds)

        results[feature] = {
            "precision": precision,
            "recall": recall,
            "fbeta": fbeta,
            "categorical_feature": cat_feature
        }

    return results

results_slice_metrics = slice_data(test, "education")
with open("../output.txt","w") as file:
    file.write(json.dumps(results_slice_metrics))


# Overall performance
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"precision, recall, fbeta: {precision}, {recall}, {fbeta}")



