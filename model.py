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
            file_path_r1 = r'G:\_Πτυχιακη\Dataset\DatasetOneFile\mergedv4.csv'
            data = pd.read_csv(file_path_r1, nrows=10)

            for i in range(2, 9):
                data = data[data.IMPROVEMENT_ITEM1 != i]

            # Drop unnecessary columns
            columns_to_drop = [
                 'POTENTIAL_ENERGY_EFFICIENCY', 'ENVIRONMENT_IMPACT_POTENTIAL',
                'ENERGY_CONSUMPTION_POTENTIAL', 'CO2_EMISS_CURR_PER_FLOOR_AREA','ENVIRONMENT_IMPACT_CURRENT','ENVIRONMENT_IMPACT_POTENTIAL', 'CO2_EMISSIONS_POTENTIAL', 'CO2_EMISSIONS_CURRENT',
                'LIGHTING_COST_POTENTIAL', 'HEATING_COST_POTENTIAL', 'HOT_WATER_COST_POTENTIAL','LIGHTING_COST_CURRENT',
                'HOT_WATER_COST_CURRENT', 'WALLS_ENERGY_EFF', 'WALLS_ENV_EFF',
                'ROOF_ENERGY_EFF', 'ROOF_ENV_EFF',  'MAINHEATC_ENV_EFF',
                'LIGHTING_ENERGY_EFF', 'LIGHTING_ENV_EFF', 'IMPROVEMENT_ITEM2', 'IMPROVEMENT_ITEM3',
                'IMPROVEMENT_ITEM1', 'IMPROVEMENT_ID2', 'IMPROVEMENT_ID3','WINDOWS_ENV_EFF','POTENTIAL_ENERGY_RATING','LMK_KEY'
            ]
            data.drop(columns=columns_to_drop, inplace=True)

            # Save the preprocessed data for potential reuse
            new_file_path = r'G:\_Πτυχιακη\Dataset\DatasetOneFile\MyAppDF.csv'
            data.to_csv(new_file_path, index=False)

            # Define the target and feature columns
            drop_columns = ['IMPROVEMENT_ID1']  # List all target column names here
            target_columns = ['IMPROVEMENT_ID1']


             # Save the feature columns for future reference
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
            file_path_r1 = r'G:\_Πτυχιακη\Dataset\DatasetOneFile\mergedv4.csv'
            data = pd.read_csv(file_path_r1, nrows=10)
            drop_columns = ['IMPROVEMENT_ID1']
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
prediction_model = PredictionModel(retrain=True)  # Change retrain to False after first run
