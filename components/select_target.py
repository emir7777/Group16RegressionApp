def get_numerical_columns(df):
    return df.select_dtypes(include=['number']).columns.tolist()