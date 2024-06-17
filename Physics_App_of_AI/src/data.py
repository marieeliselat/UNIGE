#data.py
import os
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Function to load & preprocess an image
def load_and_preprocess_image(image_path, target_size=(64, 64)):
    try:
        # Open the image file
        with Image.open(image_path) as img:
            # Calculate the cropping box coordinates
            left = (img.width - 207) // 2
            top = (img.height - 207) // 2
            right = (img.width + 207) // 2
            bottom = (img.height + 207) // 2
            # Crop image to central 207x207 pixels
            img_cropped = img.crop((left, top, right, bottom))
            # Resize image to target size
            img_resized = img_cropped.resize(target_size, Image.Resampling.LANCZOS)
            # Normalize pixel values to range [0, 1]
            img_normalized = np.array(img_resized) / 255.0
            return img_normalized
    except IOError:
        # Handle errors 
        print(f"Error opening {image_path}. Skipping.")
        return np.zeros((target_size[0], target_size[1], 3))

# Function to load labels from a CSV file
def load_labels(labels_path):
    return pd.read_csv(labels_path)

# Function to list all image files in a directory
def list_image_files(image_dir):
    return [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

# Function to generate batches of data for training/evaluation
def generate_data(image_dir, labels_df, batch_size=32, augment=False, target_columns=None):
    # Define data augmentation parameters
    if augment:
        datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    else:
        datagen = ImageDataGenerator()

    num_images = len(labels_df)
    while True:
        for start in range(0, num_images, batch_size):
            end = min(start + batch_size, num_images)
            # Get batch of image IDs
            ids_batch = labels_df['GalaxyID'][start:end]
            # Load & preprocess images in the batch
            images_batch = [load_and_preprocess_image(os.path.join(image_dir, f"{id}.jpg")) for id in ids_batch]
            # Get corresponding labels
            labels_batch = labels_df.iloc[start:end][target_columns].values if target_columns else None

            # Apply data augmentation if specified
            if augment:
                images_batch = np.array(images_batch)
                for i in range(images_batch.shape[0]):
                    images_batch[i] = datagen.random_transform(images_batch[i])

            yield np.array(images_batch), labels_batch
