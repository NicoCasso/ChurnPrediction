import pandas as pd
from typing import Callable

def exclude_spaces(column_name: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
    
    def transform(dataframe: pd.DataFrame) -> pd.DataFrame:
        new_dataframe = dataframe.copy()
        new_dataframe[column_name] = new_dataframe[column_name].replace(' ', 0).astype('float64')
        return new_dataframe
    
    return transform

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, Binarizer

# comment
def create_preprocessor() -> Pipeline :
    preprocessor = make_pipeline(
        make_column_transformer(
            (FunctionTransformer(lambda x: (x == 'Male').astype(int), validate=False), ["gender"]), # 2 values: Female, Male
            (Binarizer(), ["SeniorCitizen"]), #, 2 int64 values: 0, 1
            (FunctionTransformer(lambda x: (x == 'Yes').astype(int), validate=False), ["Partner"]), #2 values: Yes, No
            (FunctionTransformer(lambda x: (x == 'Yes').astype(int), validate=False),["Dependents"]), # 2 values: No, Yes
            (StandardScaler(),["tenure"]), # 73 int64 values.
            (FunctionTransformer(lambda x: (x == 'Yes').astype(int), validate=False),["PhoneService"]), # 2 values: No, Yes
            (OneHotEncoder(),["MultipleLines"]), # 3 values: No phone service, No, Yes
            (OneHotEncoder(),["InternetService"]), # 3 values: DSL, Fiber optic, No
            (OneHotEncoder(),["OnlineSecurity"]), #  3 values: No, Yes, No internet service
            (OneHotEncoder(),["OnlineBackup"]), # 3 values: Yes, No, No internet service
            (OneHotEncoder(),["DeviceProtection"]), # 3 values: No, Yes, No internet service        
            (OneHotEncoder(),["TechSupport"]), # 3 values: No, Yes, No internet service
            (OneHotEncoder(),["StreamingTV"]), # 3 values: No, Yes, No internet service
            (OneHotEncoder(),["StreamingMovies"]), # 3 values: No, Yes, No internet service
            (OneHotEncoder(),["Contract"]), # 3 values: Month-to-month, One year, Two year
            (FunctionTransformer(lambda x: (x == 'Yes').astype(int), validate=False),["PaperlessBilling"]), # 2 values: Yes, No
            (OneHotEncoder(),["PaymentMethod"]), # 4 values: Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)
            (StandardScaler(), ["MonthlyCharges"]), # 1585 float64 values.
            (make_pipeline(
                FunctionTransformer(exclude_spaces(column_name="TotalCharges"), validate=False),
                StandardScaler()
                ), ["TotalCharges"]), # 6531 values.
            
            #(FunctionTransformer(lambda x: (x == 'Yes').astype(int), validate=False), ["Churn"]), #, 2 values: No, Yes
            remainder='passthrough'

        )
    )
    return preprocessor


import tensorflow as tf

def build_nn_model(X_train:pd.DataFrame) -> tf.keras.Model :
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='sigmoid'),
        tf.keras.layers.Dense(16, activation='sigmoid'),
        tf.keras.layers.Dense(16, activation='sigmoid'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC', 'recall']
    )

    model.summary()

    return model

