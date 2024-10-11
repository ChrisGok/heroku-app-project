# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model is created by C. Gokus. It is a Random Forest Classifier using the default parameters in scikit-learn except number of estimators which is set to 1000.

## Intended Use

The model uses US census data in order to predict the salary based on socio economic data. Feeding in certain socia economic data, the model predicts if the salary is above or below a certain salary threshold.

## Training Data

The model is trained on US census data: https://archive.ics.uci.edu/dataset/20/census+income

The data is cleaned and transformed. The target class is “salary” . The original dataset has 32561 rows. 80% of the original dataset was used for training purposes.

## Evaluation Data

20% of the original dataset was used for testing.

## Metrics

The following metrics were used to measure the model performance:

precision: 0.72
recall: 0.63
fbeta: 0.67

## Ethical Considerations

Since the underlying data is socia economic data, like race and sex there can be potential bias in the model. However since the model is just used to predict the salary there is no danger for biased decisions based on the model output

## Caveats and Recommendations

The Random Forest model is used primarily with it’s default parameters. No hyperparameter tuning was implemented.

When dealing with socio-economic data, potential bias could also be measured and taken into consideration.