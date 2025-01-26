<<<<<<< HEAD
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def data_scaler(X_train, X_test):

    scaler=StandardScaler()
    
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    
=======
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def data_scaler(X_train, X_test):

    scaler=StandardScaler()
    
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    
>>>>>>> d7e64b796ce19bc24494cf120a3bbe8eb59697e3
    return X_train_scaled,X_test_scaled,scaler