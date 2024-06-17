import os
import pandas as pd
import numpy as np
import argparse
from src.data import load_labels, list_image_files, generate_data
from src.models import create_classification_model, create_regression_model
from src.utils import plot_training_history, get_callbacks
from sklearn.model_selection import KFold
from src.data import load_labels, list_image_files, generate_data
from src.models import create_classification_model, create_regression_model
from src.utils import plot_training_history, get_callbacks

# Function to train classification model
def train_classification():
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

    initial_epochs = 100  
    # List to store validation accuracy/loss for each fold
    val_acc_per_fold = []  
    val_loss_per_fold = []  

    # Iterate over each fold
    for train_index, val_index in kfold.split(full_df):
        # Split DataFrame into training and validation sets
        train_df = full_df.iloc[train_index]
        val_df = full_df.iloc[val_index]

        # Generate training & validation data
        train_generator = generate_data(image_dir, train_df, batch_size=32, augment=True, target_columns=['Class1.1', 'Class1.2', 'Class1.3'])
        val_generator = generate_data(image_dir, val_df, batch_size=32, target_columns=['Class1.1', 'Class1.2', 'Class1.3'])

        # Create classification model
        model = create_classification_model()

        # Train model
        print(f"Training fold {fold_no} for classification...")
        history = model.fit(
            train_generator,
            steps_per_epoch=len(train_df) // 32,
            validation_data=val_generator,
            validation_steps=len(val_df) // 32,
            epochs=initial_epochs,
            callbacks=get_callbacks()  
        )

        # Evaluate model on validation set
        val_loss, val_acc = model.evaluate(val_generator, steps=len(val_df) // 32)
        print(f"Fold {fold_no} - Validation Accuracy: {val_acc}, Validation Loss: {val_loss}")

        # Store validation accuracy and loss
        val_acc_per_fold.append(val_acc)
        val_loss_per_fold.append(val_loss)

        # Plot training history for each fold
        model_dir = './models/classification' 
        os.makedirs(model_dir, exist_ok=True)
        plot_training_history(history, fold_no, 'classification', model_dir)

        # Increment fold counter
        fold_no += 1  

    # Summary statistics for validation accuracy & loss
    mean_val_acc = np.mean(val_acc_per_fold)
    mean_val_loss = np.mean(val_loss_per_fold)
    std_val_acc = np.std(val_acc_per_fold)
    std_val_loss = np.std(val_loss_per_fold)

    # Save results to file
    results_path = os.path.join(model_dir, 'cross_validation_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"Validation Accuracy per fold: {val_acc_per_fold}\n")
        f.write(f"Validation Loss per fold: {val_loss_per_fold}\n")
        f.write(f"Mean Validation Accuracy: {mean_val_acc}\n")
        f.write(f"Mean Validation Loss: {mean_val_loss}\n")
        f.write(f"Standard Deviation of Validation Accuracy: {std_val_acc}\n")
        f.write(f"Standard Deviation of Validation Loss: {std_val_loss}\n")

    # Print statistics
    print(f"Validation Accuracy per fold: {val_acc_per_fold}")
    print(f"Validation Loss per fold: {val_loss_per_fold}")
    print(f"Mean Validation Accuracy: {mean_val_acc}")
    print(f"Mean Validation Loss: {mean_val_loss}")
    print(f"Standard Deviation of Validation Accuracy: {std_val_acc}")
    print(f"Standard Deviation of Validation Loss: {std_val_loss}")

    print("Classification training completed.")

# Function to train regression models for Exercise 2
def train_regression():
    # Paths to label file & image directory
    labels_path = 'data/labels.csv'  
    image_dir = 'data/images' 

    # Load labels & list of image files
    labels_df = load_labels(labels_path)
    image_files = list_image_files(image_dir)

    # Add 'filename' column to labels DataFrame for merging
    labels_df['filename'] = labels_df['GalaxyID'].apply(lambda x: f"{x}.jpg")
    full_df = pd.merge(pd.DataFrame({'filename': image_files}), labels_df, on='filename')

    # Target columns for regression tasks - Exercise 2 
    target_columns_list = [['Class2.1', 'Class2.2'], ['Class7.1', 'Class7.2', 'Class7.3']]

    # Iterate over each target columns set
    for target_columns in target_columns_list:
        # K-Fold Cross Validation with 5 splits
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_no = 1  
        initial_epochs = 100   
        # List to store validation MAE/loss for each fold
        val_mae_per_fold = []  
        val_loss_per_fold = []  

        # Iterate over each fold
        for train_index, val_index in kfold.split(full_df):
            # Split the DataFrame into training & validation sets
            train_df = full_df.iloc[train_index]
            val_df = full_df.iloc[val_index]

            # Generate training & validation data
            train_generator = generate_data(image_dir, train_df, batch_size=32, augment=True, target_columns=target_columns)
            val_generator = generate_data(image_dir, val_df, batch_size=32, target_columns=target_columns)

            # Create regression model
            model = create_regression_model(len(target_columns))

            # Train model
            print(f"Training fold {fold_no} for regression target columns {target_columns}...")
            history = model.fit(
                train_generator,
                steps_per_epoch=len(train_df) // 32,
                validation_data=val_generator,
                validation_steps=len(val_df) // 32,
                epochs=initial_epochs,
                callbacks=get_callbacks()  
            )
            # Evaluate model on validation set
            val_loss, val_mae = model.evaluate(val_generator, steps=len(val_df) // 32)
            print(f"Fold {fold_no} - Validation MAE: {val_mae}, Validation Loss: {val_loss}")

            # Store validation MAE & loss
            val_mae_per_fold.append(val_mae)
            val_loss_per_fold.append(val_loss)

            # Plot training history for each fold
            model_dir = f'./models/regression'  
            os.makedirs(model_dir, exist_ok=True)
            plot_training_history(history, fold_no, 'regression', model_dir)

            fold_no += 1  

        # Summary statistics for validation MAE & loss
        mean_val_mae = np.mean(val_mae_per_fold)
        mean_val_loss = np.mean(val_loss_per_fold)
        std_val_mae = np.std(val_mae_per_fold)
        std_val_loss = np.std(val_loss_per_fold)

        # Save results 
        results_path = os.path.join(model_dir, 'cross_validation_results.txt')
        with open(results_path, 'w') as f:
            f.write(f"Validation MAE per fold: {val_mae_per_fold}\n")
            f.write(f"Validation Loss per fold: {val_loss_per_fold}\n")
            f.write(f"Mean Validation MAE: {mean_val_mae}\n")
            f.write(f"Mean Validation Loss: {mean_val_loss}\n")
            f.write(f"Standard Deviation of Validation MAE: {std_val_mae}\n")
            f.write(f"Standard Deviation of Validation Loss: {std_val_loss}\n")

        # Print statistics
        print(f"Validation MAE per fold for {target_columns}: {val_mae_per_fold}")
        print(f"Validation Loss per fold for {target_columns}: {val_loss_per_fold}")
        print(f"Mean Validation MAE for {target_columns}: {mean_val_mae}")
        print(f"Mean Validation Loss for {target_columns}: {mean_val_loss}")
        print(f"Standard Deviation of Validation MAE for {target_columns}: {std_val_mae}")
        print(f"Standard Deviation of Validation Loss for {target_columns}: {std_val_loss}")

    print("Regression training completed.")

# Function to train the regression model for Exercise 3
def train_regression_ex3():
    # Paths to label file & image directory
    labels_path = 'data/labels.csv'  
    image_dir = 'data/images' 

    # Load labels & list of image files
    labels_df = load_labels(labels_path)
    image_files = list_image_files(image_dir)

    # Add 'filename' column to labels DataFrame for merging
    labels_df['filename'] = labels_df['GalaxyID'].apply(lambda x: f"{x}.jpg")
    full_df = pd.merge(pd.DataFrame({'filename': image_files}), labels_df, on='filename')

    # Target columns for regression tasks - Exercise 3
    target_columns = ['Class6.1', 'Class6.2', 'Class8.1', 'Class8.2', 'Class8.3', 'Class8.4', 'Class8.5', 'Class8.6', 'Class8.7']

    # K-Fold Cross Validation with 5 splits
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_no = 1  
    initial_epochs = 100  
    # List to store validation MAE/loss for each fold
    val_mae_per_fold = []  
    val_loss_per_fold = [] 

    # Iterate over each fold
    for train_index, val_index in kfold.split(full_df):
        # Split DataFrame into training & validation sets
        train_df = full_df.iloc[train_index]
        val_df = full_df.iloc[val_index]

        # Generate training & validation data
        train_generator = generate_data(image_dir, train_df, batch_size=32, augment=True, target_columns=target_columns)
        val_generator = generate_data(image_dir, val_df, batch_size=32, target_columns=target_columns)

        # Create regression model
        model = create_regression_model(len(target_columns))

        # Train model
        print(f"Training fold {fold_no} for regression exercise 3 target columns {target_columns}...")
        history = model.fit(
            train_generator,
            steps_per_epoch=len(train_df) // 32,
            validation_data=val_generator,
            validation_steps=len(val_df) // 32,
            epochs=initial_epochs,
            callbacks=get_callbacks()  # Callbacks for early stopping, learning rate reduction, etc.
        )

        # Evaluate model on validation set
        val_loss, val_mae = model.evaluate(val_generator, steps=len(val_df) // 32)
        print(f"Fold {fold_no} - Validation MAE: {val_mae}, Validation Loss: {val_loss}")

        # Store validation MAE & loss
        val_mae_per_fold.append(val_mae)
        val_loss_per_fold.append(val_loss)

        # Plot training history for each fold
        model_dir = './models/regression_ex3'  
        os.makedirs(model_dir, exist_ok=True)
        plot_training_history(history, fold_no, 'regression_ex3', model_dir)

        fold_no += 1  

    # Summary statistics for validation MAE & loss
    mean_val_mae = np.mean(val_mae_per_fold)
    mean_val_loss = np.mean(val_loss_per_fold)
    std_val_mae = np.std(val_mae_per_fold)
    std_val_loss = np.std(val_loss_per_fold)

    # Save results 
    results_path = os.path.join(model_dir, 'cross_validation_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"Validation MAE per fold: {val_mae_per_fold}\n")
        f.write(f"Validation Loss per fold: {val_loss_per_fold}\n")
        f.write(f"Mean Validation MAE: {mean_val_mae}\n")
        f.write(f"Mean Validation Loss: {mean_val_loss}\n")
        f.write(f"Standard Deviation of Validation MAE: {std_val_mae}\n")
        f.write(f"Standard Deviation of Validation Loss: {std_val_loss}\n")

    # Print statistics
    print(f"Validation MAE per fold for {target_columns}: {val_mae_per_fold}")
    print(f"Validation Loss per fold for {target_columns}: {val_loss_per_fold}")
    print(f"Mean Validation MAE for {target_columns}: {mean_val_mae}")
    print(f"Mean Validation Loss for {target_columns}: {mean_val_loss}")
    print(f"Standard Deviation of Validation MAE for {target_columns}: {std_val_mae}")
    print(f"Standard Deviation of Validation Loss for {target_columns}: {std_val_loss}")

    print("Regression training for Exercise 3 completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train models for classification and regression tasks.')
    parser.add_argument('--task', type=str, required=True, help='Task to run (classification, regression, regression_ex3)', choices = ['classification', 'regression', 'regression_ex3'])
    args = parser.parse_args()

    if args.task == 'classification':
        train_classification()
    elif args.task == 'regression':
        train_regression()
    elif args.task == 'regression_ex3':
        train_regression_ex3()
    else:
        print("Invalid task. Please choose from classification, regression, or regression_ex3.")