import pandas as pd 
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

import gc 
gc.enable()

class DataProcessor:

    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.data = None
        self.to_scale = []
        self.categorical_cols = ['device_os', 'source', 'payment_type', 'employment_status']
        self.cols_to_del = ['housing_status', 'device_fraud_count', 'bank_branch_count_8w','month', 'prev_address_months_count', 'bank_months_count', 'days_since_request', 'proposed_credit_limit']
        self.col_with_nan = ['current_address_months_count']
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None

    def load_data(self):
        self.data = pd.read_csv(self.csv_file_path)
        self.data.drop(self.cols_to_del, axis=1, inplace=True)
        self.data.loc[:, self.col_with_nan] = self.data[self.col_with_nan].replace(-1, np.nan)
    
    def knn_impute(self):
        knn_imputer = KNNImputer()
        self.data[self.col_with_nan] = knn_imputer.fit_transform(self.data[self.col_with_nan])
        self.data[self.col_with_nan] = self.data[self.col_with_nan].astype(int)

    def scale_data(self):
        scl = MinMaxScaler()
        self.to_scale = [col for col in self.to_scale.col if col not in ['device_os', 'source', 'payment_type', 'fraud_bool', 'employment_status']]
        for col in self.to_scale:
            self.data[col] = scl.fit_transform(self.data[col].values.reshape(-1, 1))

    def  one_hot_encode(self):
        self.data = pd.get_dummies(self.data, columns=self.categorical_cols, prefix=self.categorical_cols)

    
    def split_data(self):
        X = self.data.drop('fraud_bool', axis=1)
        y = self.data['fraud_bool'].copy()

        # Load your data and split it into train, validation, and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.2, random_state=42)
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

data_processor = DataProcessor('Base.csv')
# Perform the steps
data_processor.load_data()
data_processor.knn_impute()
data_processor.scale_data()
data_processor.one_hot_encode()
data_processor.split_data()

# Access the processed data and splits
X_train = data_processor.X_train
X_val = data_processor.X_val
y_train = data_processor.y_train
y_val = data_processor.y_val
X_test = data_processor.X_train
y_test = data_processor.y_test