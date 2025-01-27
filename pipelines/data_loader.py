import pandas as pd

def data_loader(file_path):
    print("Loading data...")
    data = pd.read_csv(file_path)
    return data
