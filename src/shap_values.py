import shap
from deeplmodel import X_test, X_train, categorical_columns, numerical_columns, model, clf
import numpy as np

X_test_dense = X_test.toarray() if hasattr(X_test, "toarray") else X_test
X_train_dense = X_train.toarray() if hasattr(X_train, "toarray") else X_train

explainer = shap.DeepExplainer(model, X_train_dense[:100])  
shap_values = explainer.shap_values(X_test_dense[:50])  
ohe = clf.named_steps['preprocessor'].named_transformers_['one-hot-encoder']
encoded_cat_names = ohe.get_feature_names_out(categorical_columns)
all_feature_names = np.concatenate([encoded_cat_names, np.array(numerical_columns)])



shap_values_squeezed = np.squeeze(shap_values, axis=2) 
shap.summary_plot(shap_values_squeezed, X_test_dense[:50], feature_names=all_feature_names)