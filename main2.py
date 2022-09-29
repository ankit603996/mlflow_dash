#https://mlflow.org/docs/latest/models.html#environment-management-tools
import xgboost
import shap
import mlflow
from sklearn.model_selection import train_test_split
import pandas as pd
# load UCI Adult Data Set; segment it into training and test sets
#X, y = shap.datasets.adult()
df = pd.read_csv('headbrain11.csv')
df.columns = ['Head_size', 'Brain_weight']
factor = 2.2
df['Head_size2'] = df['Head_size'] + factor
# model = sm.ols(formula='Head_size ~ Brain_weight', data=df).fit()
# print(model.summary())

X = df.drop("Head_size", axis=1)
y = df.Head_size

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# train XGBoost model
model = xgboost.XGBClassifier().fit(X_train, y_train)

# construct an evaluation dataset from the test set
eval_data = X_test
eval_data["label"] = y_test

with mlflow.start_run() as run:
    model_info = mlflow.sklearn.log_model(model, "model")
    result = mlflow.evaluate(
        model_info.model_uri,
        eval_data,
        targets="label",
        model_type="classifier",
        dataset_name="adult",
        evaluators=["default"],
    )