import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import cloudpickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import shap


data = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
data['TotalCharges'] = data['TotalCharges'].replace(' ', 0)

y = [1 if x == 'Yes' else 0 for x in data['Churn']]
X = data.drop(columns=['customerID','Churn'])

numerical_columns = ['SeniorCitizen','tenure', 'MonthlyCharges', 'TotalCharges']
categorical_columns = ['gender', 'Partner', 'Dependents', 
        'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod']

categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
numerical_preprocessor = StandardScaler()

preprocessor = ColumnTransformer([
    ('one-hot-encoder', categorical_preprocessor, categorical_columns),
    ('standard-scaler', numerical_preprocessor, numerical_columns)])

clf = Pipeline(
    steps=[("preprocessor", preprocessor)]
)

X = clf.fit_transform(X)

X_train_0, X_test, y_train_0, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


X_train, X_val, y_train, y_val = train_test_split(
    X_train_0, y_train_0, test_size=0.2, random_state=42, stratify=y_train_0
)

def build_model():
    
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

    return model

model = build_model()

X_train = X_train.toarray() if hasattr(X_train, "toarray") else X_train
X_val = X_val.toarray() if hasattr(X_val, "toarray") else X_val

y_train = np.array(y_train)
y_val = np.array(y_val)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=16,
    verbose=1
)

def plot_loss_acc(history, validation=True):
    """
    Trace la loss et l'accuracy du modèle pendant l'entraînement.
    """
    plt.figure(figsize=(15, 5))

    # Loss
    plt.subplot(1, 4, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    if validation and 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Évolution de la Loss')
    plt.xlabel('Époque')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    # Accuracy
    plt.subplot(1, 4, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    if validation and 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title("Évolution de l'Accuracy")
    plt.xlabel('Époque')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    # AUC
    plt.subplot(1, 4, 3)
    plt.plot(history.history['AUC'], label='Train AUC')
    if validation and 'val_AUC' in history.history:
        plt.plot(history.history['val_AUC'], label='Val AUC')
    plt.title("Évolution de AUC")
    plt.xlabel('Époque')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid()

    # Recall
    plt.subplot(1, 4, 4)
    plt.plot(history.history['recall'], label='Train recall')
    if validation and 'val_recall' in history.history:
        plt.plot(history.history['val_recall'], label='Val recall')
    plt.title("Évolution de recall")
    plt.xlabel('Époque')
    plt.ylabel('recall')
    plt.legend()
    plt.grid()


    plt.tight_layout()
    plt.savefig('img/Evolution_of_training')
    
plot_loss_acc(history)
plt.close()

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',    # surveille la perte de validation
    patience=3,            # tolère 3 époques sans amélioration
    restore_best_weights=True
)


X_test = X_test.toarray() if hasattr(X_test, "toarray") else X_test
y_test = np.array(y_test)

test_results = model.evaluate(X_test, y_test, verbose=1)

# print(test_results)

# Predict probabilities
y_pred_probs = model.predict(X_test)

# Binary classification threshold
y_pred = (y_pred_probs >= 0.5).astype(int)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
plt.title("Matrice de Confusion")
plt.xlabel("Classe Prédite")
plt.ylabel("Classe Réelle")
plt.savefig('img/Matrice_de_Confusion.png')
plt.close()

X_test_dense = X_test.toarray() if hasattr(X_test, "toarray") else X_test
X_train_dense = X_train.toarray() if hasattr(X_train, "toarray") else X_train

explainer = shap.DeepExplainer(model, X_train_dense[:100])  
shap_values = explainer.shap_values(X_test_dense[:50])  
ohe = clf.named_steps['preprocessor'].named_transformers_['one-hot-encoder']
encoded_cat_names = ohe.get_feature_names_out(categorical_columns)
all_feature_names = np.concatenate([encoded_cat_names, np.array(numerical_columns)])

shap_values_squeezed = np.squeeze(shap_values, axis=2) 
shap.summary_plot(shap_values_squeezed, X_test_dense[:50], feature_names=all_feature_names, show=False)
plt.savefig("img/shap_summary_plot.png", bbox_inches='tight', dpi=300)

with open('model/model.pkl', 'wb') as f:
    pkl.dump(model,f)