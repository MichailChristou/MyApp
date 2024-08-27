import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os


class PredictionModel:
    def __init__(self, retrain=False):
        model_path = "trained_model.pkl"
        self.columns = None

        if retrain or not os.path.exists(model_path):
            # Load and preprocess the dataset
            file_path_r1 = r'E:\Gorillino\Unipi\_Πτυχιακη\Dataset\DatasetOneFile\MyAppDS.csv'
            data = pd.read_csv(file_path_r1, nrows=10000)

            columns_to_drop = ['LMK_KEY']
            data.drop(columns=columns_to_drop, inplace=True)

            drop_columns = ['IMPROVEMENT_ITEM1']
            target_columns = ['IMPROVEMENT_ITEM1']

            self.columns = data.drop(columns=drop_columns).columns
            self.default_values = {
                'LMK_KEY': 'default_value'
            }

            # Split the data into training and testing sets
            X = data.drop(columns=drop_columns)
            y = data[target_columns].values.ravel()  # Flatten y to a 1D array

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            # Train the model
            self.model = RandomForestClassifier()
            self.model.fit(X_train, y_train)

            # Save the model to disk
            joblib.dump(self.model, model_path)
            print("Model trained and saved to disk.")

            # Calculate accuracy on test set
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print("Model training completed. Accuracy:", accuracy)

        else:
            # Load the model if it already exists
            self.model = joblib.load(model_path)
            print("Model loaded from disk.")

            # Load the dataset to get the column names, but only need a few rows
            file_path_r1 = r'E:\Gorillino\Unipi\_Πτυχιακη\Dataset\DatasetOneFile\MyAppDS.csv'
            data = pd.read_csv(file_path_r1, nrows=10)
            drop_columns = ['IMPROVEMENT_ITEM1']
            self.columns = data.drop(columns=drop_columns).columns

    def predict(self, input_data):
        input_df = pd.DataFrame(input_data)
        prediction = self.model.predict(input_df)
        return prediction[0]

    def get_columns(self):
        return self.columns

    def get_default_value(self, column):
        return self.default_values.get(column, None)


# Create an instance of the model
# Set `retrain=True` if you want to retrain the model
prediction_model = PredictionModel(retrain=False)  # Change retrain to False after first run
