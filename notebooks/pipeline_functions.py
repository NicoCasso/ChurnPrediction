import pandas as pd

def exclude_spaces(dataframe : pd.DataFrame, column_name: str ) -> pd.DataFrame:
    new_dataframe = dataframe.copy()
    new_dataframe[column_name] = new_dataframe[column_name].replace(' ', 0).astype('float64')
    return new_dataframe