import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os

def data_preprocessor(data):
    """
    Cleans and preprocesses the dataset.
    """
    print("Preprocessing data...")

##################################

    data['location'] = data['location'].fillna('Unknown')
    data['size'] = data['size'].fillna(data['size'].mode()[0])
    data['bath'] = data['bath'].fillna(data['bath'].median())
    data['balcony'] = data['balcony'].fillna(data['balcony'].median())
    #data['society'] = data['society'].fillna(data['society'].mode()[0])

    # Drop unnecessary columns
    if 'availability' in data.columns:
        data = data.drop(['availability'], axis=1)
    if 'society' in data.columns:
        data = data.drop(['society'], axis=1)

    # Extract numeric value from the 'size' column
    data['size'] = data['size'].str.extract(r'(\d+)').fillna(0).astype(int)

    # Convert 'total_sqft' to numeric
    def convert_sqft_to_num(sqft):
        try:
            if '-' in sqft:
                sqft_range = sqft.split('-')
                return (float(sqft_range[0]) + float(sqft_range[1])) / 2
            return float(sqft)
        except:
            return None

    # Convert 'total_sqft' to numeric
    data['total_sqft'] = data['total_sqft'].apply(convert_sqft_to_num)
    data = data.dropna(subset=['total_sqft'])

    # Remove outliers
    #upper_limit_sqft = data['total_sqft'].quantile(0.99)
    #upper_limit_bath = data['bath'].quantile(0.99)
    #upper_limit_price = data['price'].quantile(0.99)

    #data['total_sqft'] = np.where(data['total_sqft'] > upper_limit_sqft, upper_limit_sqft, data['total_sqft'])
    #data['bath'] = np.where(data['bath'] > upper_limit_bath, upper_limit_bath, data['bath'])
    #data['price'] = np.where(data['price'] > upper_limit_price, upper_limit_price, data['price'])

    # One-hot encode categorical features
    #categorical_features = ['location', 'area_type', 'society', 'size']
    categorical_features = ['location', 'area_type',  'size']

    data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

    # Feature engineering
    #data['price_per_sqft'] = data['price'] / data['total_sqft']
    data['log_price'] = np.log1p(data['price'])

    # Separate features and target
    X = data.drop(['price', 'log_price'], axis=1)
    y = data['log_price']

    feature_names = X.columns

>>>>>>> d7e64b796ce19bc24494cf120a3bbe8eb59697e3
    return X, y,feature_names
