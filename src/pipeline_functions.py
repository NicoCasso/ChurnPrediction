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

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
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

def build_nn_model(X_train:pd.DataFrame) -> tf.keras.Model :
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
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
            tf.keras.metrics.Precision(name='precision'),
            #tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='roc_auc')
            #F1Score()  # Custom metric
        ]
    )

    model.summary()

    return model

