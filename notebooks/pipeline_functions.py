import pandas as pd
from typing import Callable

def exclude_spaces(column_name: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
    
    def transform(dataframe: pd.DataFrame) -> pd.DataFrame:
        new_dataframe = dataframe.copy()
        new_dataframe[column_name] = new_dataframe[column_name].replace(' ', 0).astype('float64')
        return new_dataframe
    
    return transform