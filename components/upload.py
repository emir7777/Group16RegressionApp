import pandas as pd
def upload_file(file_path):
    df = pd.read_csv(file_path)
    return df