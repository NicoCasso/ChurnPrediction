import pandas as pd
import numpy as np
from typing import Callable

#______________________________________________________________________________
#
# region remplace les espaces par '0'
#______________________________________________________________________________
def exclude_spaces_from_totalCharges(dataframe : pd.DataFrame) -> pd.DataFrame:

    new_dataframe = dataframe.copy()
    new_dataframe['TotalCharges'] = (
        new_dataframe['TotalCharges'].astype(str).str.replace(' ', '', regex=False)
        .replace('', '0').astype(float)
    )
    return new_dataframe
    

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, Binarizer

# Fonctions nommées (au lieu des lambda)
def encode_male(x):
    return (x == 'Male').astype(int)

def encode_yes(x):
    return (x == 'Yes').astype(int)

#______________________________________________________________________________
#
# region create_preprocessor
#______________________________________________________________________________
def create_preprocessor() -> Pipeline :
    preprocessor = make_pipeline(
        make_column_transformer(
            (FunctionTransformer(encode_male, validate=False), ["gender"]), # 2 values: Female, Male
            (Binarizer(), ["SeniorCitizen"]), #, 2 int64 values: 0, 1
            (FunctionTransformer(encode_yes, validate=False), ["Partner"]), #2 values: Yes, No
            (FunctionTransformer(encode_yes, validate=False),["Dependents"]), # 2 values: No, Yes
            (StandardScaler(),["tenure"]), # 73 int64 values.
            (FunctionTransformer(encode_yes, validate=False),["PhoneService"]), # 2 values: No, Yes
            (OneHotEncoder(),["MultipleLines"]), # 3 values: No phone service, No, Yes
            (OneHotEncoder(),["InternetService"]), # 3 values: DSL, Fiber optic, No
            (OneHotEncoder(),["OnlineSecurity"]), #  3 values: No, Yes, No internet service
            (OneHotEncoder(),["OnlineBackup"]), # 3 values: Yes, No, No internet service
            (OneHotEncoder(),["DeviceProtection"]), # 3 values: No, Yes, No internet service        
            (OneHotEncoder(),["TechSupport"]), # 3 values: No, Yes, No internet service
            (OneHotEncoder(),["StreamingTV"]), # 3 values: No, Yes, No internet service
            (OneHotEncoder(),["StreamingMovies"]), # 3 values: No, Yes, No internet service
            (OneHotEncoder(),["Contract"]), # 3 values: Month-to-month, One year, Two year
            (FunctionTransformer(encode_yes, validate=False),["PaperlessBilling"]), # 2 values: Yes, No
            (OneHotEncoder(),["PaymentMethod"]), # 4 values: Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)
            (StandardScaler(), ["MonthlyCharges"]), # 1585 float64 values.
            (make_pipeline(
                FunctionTransformer(exclude_spaces_from_totalCharges, validate=False),
                StandardScaler()
                ), ["TotalCharges"]), # 6531 values.
            
            #(FunctionTransformer(encode_yes, validate=False), ["Churn"]), #, 2 values: No, Yes
            remainder='passthrough'

        )
    )
    return preprocessor


import tensorflow as tf


#______________________________________________________________________________
#
# region metrique spécifique
#______________________________________________________________________________
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1-score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.round(y_pred)

        tp = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32))
        fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, tf.float32))
        fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), tf.float32))

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        precision = self.tp / (self.tp + self.fp + tf.keras.backend.epsilon())
        recall = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
        return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

    def reset_states(self):
        self.tp.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)

#______________________________________________________________________________
#
# region construction du modèle
#______________________________________________________________________________
def build_nn_model(input_shape : tuple[int, int]) -> tf.keras.Model :
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(64, activation='sigmoid'),
        tf.keras.layers.Dense(32, activation='sigmoid'),
        tf.keras.layers.Dense(16, activation='sigmoid'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            #tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            F1Score(),  # Custom metric
            tf.keras.metrics.AUC(name='roc_auc')
      
        ]
    )

    return model

