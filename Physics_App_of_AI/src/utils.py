#utils.py
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

#Function to plot training history
def plot_training_history(history, fold_no, task_name, model_dir):
    plt.figure(figsize=(12, 6))

    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'Model Accuracy for Fold {fold_no}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')

    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Model Loss for Fold {fold_no}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    # Save plot to specified directory
    plt.savefig(os.path.join(model_dir, f'{task_name}_training_history_fold_{fold_no}.png'))
    plt.close()

#Function to get an idea of what's happening in the model
def get_callbacks():
    # Stop training when validation loss stops improving
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # Reduce learning rate when validation loss plateaus
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    # Adjust learning rate according to cosine annealing schedule
    learning_rate_scheduler = LearningRateScheduler(lambda epoch, lr: lr * (np.cos(np.pi * epoch / 100) + 1) / 2)
    return [early_stopping, reduce_lr, learning_rate_scheduler]
