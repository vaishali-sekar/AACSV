
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def data_scaler(X_train, X_test):

    scaler=StandardScaler()
    
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    return X_train_scaled,X_test_scaled,scaler
