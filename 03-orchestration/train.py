from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import pickle
import os


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
    vectorizer = DictVectorizer(sparse=True)
    feature_matrix = vectorizer.fit_transform(X_train)
    model = LinearRegression()
    model.fit(feature_matrix, y_train)
    print(model.intercept_)
    return model.intercept_
