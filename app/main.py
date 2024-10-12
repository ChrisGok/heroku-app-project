import os
import pandas as pd
import pickle
from pydantic import BaseModel, Field
from fastapi import FastAPI
from ml.data import process_data
from ml.model import inference


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

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

file_path_base = os.path.join(os.getcwd(),'model')

model = pickle.load(open(f'{file_path_base}/classifier.pkl', 'rb'))
encoder = pickle.load(open(f'{file_path_base}/encoder.pkl', 'rb'))
lb = pickle.load(open(f'{file_path_base}/lb.pkl', 'rb'))

class Data(BaseModel):
    age: int
    workclass: str 
    fnlgt: int
    education: str
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

# provide example how the request body should look

    class ConfigDict:
        json_schema_extra = {
            "examples": {
                    'age': 25,
                    'workclass': 'Private',
                    'fnlgt': 215646,
                    'education': 'Bachelors',
                    'education_num': 10,
                    'marital_status': 'Never-married',
                    'occupation': 'Handlers-cleaners',
                    'relationship': 'Not-in-family',
                    'race': 'White',
                    'sex': 'Female',
                    'capital_gain': 100,
                    'capital_loss': 50,
                    'hours_per_week': 25,
                    'native_country': 'United-States'
            }
        }

app = FastAPI(
    title="Census data ML model API",
    description="An API to push data and get ML model result",
    version="1.0.0"
)

@app.get("/")
async def welcome():
    return "Heyho! Nice to see you!"

@app.post("/prediction")
async def model_prediction(input: Data):
    
    print(input)
    print(type(input))
    input_data = input.dict(by_alias=True)
    input_data = pd.DataFrame(data=input_data, index=[0])
    # input_data = pd.DataFrame(input.dict(by_alias=True), index=[0])
    print(input_data)

    X_test, _, _, _ = process_data(
        input_data, categorical_features=cat_features, training=False, encoder=encoder, lb=lb
        )
    
    preds = inference(model, X_test).tolist()
    print(preds,preds[0], type(preds))

    if preds[0] == 0:
       sal_pred =  "<=50k"
    else:
       sal_pred = ">50k"

    return {'prediction': sal_pred}

