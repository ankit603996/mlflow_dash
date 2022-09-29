import numpy as np
import pandas as pd
import statsmodels.api as sm
from dash.exceptions import PreventUpdate
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash import Dash, Input, Output, callback, dash_table
import mlflow
import mlflow.statsmodels
#https://www.youtube.com/watch?v=r0do1KVEGqM
#pip install mlflow[extra]
#https://www.youtube.com/watch?v=EbIEk0DB-H8
#mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 --port 5000 --env-manager=conda
if __name__=="__main__":
#    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name="mlflow_dash_learning1")
    df = pd.read_csv('headbrain11.csv')
    df.columns = ['Head_size', 'Brain_weight']
    factor = 2.2
    df['Head_size2'] = df['Head_size']+factor
    #model = sm.ols(formula='Head_size ~ Brain_weight', data=df).fit()
    #print(model.summary())

    X = df.drop("Head_size",axis=1)
    y = df.Head_size
    feature_names = X.columns.to_list()
    from sklearn.preprocessing import StandardScaler

    standizer = StandardScaler()
    X = standizer.fit_transform(X)

    import statsmodels.api as sm
    exog = sm.add_constant(X)
    poission_model = sm.GLM(y, exog, family=sm.families.Poisson())
    result = poission_model.fit()
    print(result.summary())
    mlflow.log_metric("aic",result.aic)
    mlflow.statsmodels.log_model(result,"poissonmodel")
#    mlflow.statsmodels.autolog(log_models=True, disable=False, exclusive=False, disable_for_unsupported_versions=False,
#                               silent=False, registered_model_name=None)
