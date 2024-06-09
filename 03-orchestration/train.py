from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import pickle
import os
import mlflow
from sklearn.metrics import mean_squared_error



if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    X = data[['PULocationID', 'DOLocationID']]
    y = data['duration']
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train = X_train.to_dict(orient='records')
    X_test = X_test.to_dict(orient='records')
    with mlflow.start_run() as run:
        vectorizer = DictVectorizer(sparse=True)
        feature_matrix = vectorizer.fit_transform(X_train)
        feature_matrix_test = vectorizer.transform(X_test)
        model = LinearRegression()
        model.fit(feature_matrix, y_train)
        ######## Predict in test dataset #################
        predictions = model.predict(feature_matrix_test)
        mse = mean_squared_error(y_test, predictions)
        ######## Loging artifacts  ######################
        mlflow.sklearn.log_model(model, "lr_model")

        with open("vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
        mlflow.log_artifact("vectorizer.pkl")

        #mlflow.sklearn.log(vectorizer, "vectorizer")
        mlflow.log_param("intercept", model.intercept_)
        mlflow.log_param("coefficients", model.coef_)
        mlflow.log_metric("mse", mse)
        ######## Register model  ######################
        model_uri = f"runs:/{run.info.run_id}/lr_model"
        mlflow.register_model(model_uri, "LinearRegressionModelHomework3")