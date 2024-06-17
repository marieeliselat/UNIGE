#eval.py
import os
import numpy as np
import pandas as pd
import argparse
from src.data import load_labels, list_image_files, generate_data
from src.models import create_classification_model, create_regression_model
from src.utils import plot_training_history, get_callbacks
from tensorflow.keras.models import load_model
from src.data import load_labels, list_image_files, generate_data
from sklearn.model_selection import KFold

# Function to evaluate model based on the specified task & target columns
def evaluate_model(model_dir, task, target_columns):
    # Paths to label file & image directory
    labels_path = 'data/labels.csv'
    image_dir = 'data/images'

    # Load labels & list of image files
    labels_df = load_labels(labels_path)
    image_files = list_image_files(image_dir)

    # Add 'filename' column to labels DataFrame for merging
    labels_df['filename'] = labels_df['GalaxyID'].apply(lambda x: f"{x}.jpg")
    full_df = pd.merge(pd.DataFrame({'filename': image_files}), labels_df, on='filename')

    # Set up 5-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_no = 1

    val_scores = []

    # Perform cross-validation
    for train_index, val_index in kfold.split(full_df):
        val_df = full_df.iloc[val_index]

        # Generate validation data
        val_generator = generate_data(image_dir, val_df, batch_size=32, target_columns=target_columns)
        model_path = os.path.join(model_dir, f'model_fold_{fold_no}.h5')
        
        # Check if model for current fold exists
        if not os.path.exists(model_path):
            print(f"Model for fold {fold_no} not found at {model_path}")
            continue

        # Load model
        model = load_model(model_path)
        
        # Evaluate model on validation data
        val_loss, val_score = model.evaluate(val_generator, steps=len(val_df) // 32)
        val_scores.append(val_score)

        fold_no += 1

    # Print validation scores and their statistics
    print(f"Validation Scores: {val_scores}")
    print(f"Mean Validation Score: {np.mean(val_scores)}")
    print(f"Standard Deviation of Validation Score: {np.std(val_scores)}")

# Main script execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate models for classification and regression tasks.')
    parser.add_argument('--task', type=str, required=True, help='Task to evaluate (classification, regression, regression_ex3)', choices = ['classification', 'regression', 'regression_ex3'])
    parser.add_argument('--target_columns', type=str, help='Target columns for regression tasks (e.g., Class2.1,Class2.2)')
    args = parser.parse_args()
    
    if args.task == 'classification':
        evaluate_model('./models/classification', args.task, ['Class1.1', 'Class1.2', 'Class1.3'])
    elif args.task == 'regression':
        if args.target_columns:
            target_columns = args.target_columns.split(',')
            evaluate_model(f'./models/regression_{target_columns}', args.task, target_columns)
        else:
            print("Target columns required for regression task.")
    elif args.task == 'regression_ex3':
        evaluate_model('./models/regression_ex3', args.task, ['Class6.1', 'Class6.2', 'Class8.1', 'Class8.2', 'Class8.3', 'Class8.4', 'Class8.5', 'Class8.6', 'Class8.7'])
    else:
        print("Invalid task. Please choose from classification, regression, or regression_ex3.")