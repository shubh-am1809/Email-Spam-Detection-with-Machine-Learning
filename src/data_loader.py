import pandas as pd

def load_data():
    df = pd.read_csv("data/spam.csv", encoding="latin-1")
    
    # Keep only required columns
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

    return df
