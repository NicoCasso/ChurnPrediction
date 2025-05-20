import matplotlib.pyplot as plt
from deeplmodel import history

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
    plt.show()
    
plot_loss_acc(history)