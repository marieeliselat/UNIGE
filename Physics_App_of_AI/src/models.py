#models.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import tensorflow as tf

# Function to create Convolutional Neural Network (CNN) model for classification
def create_classification_model():
    # Sequential model builds layer by layer
    model = Sequential([
        # 1st convolutional layer with 32 filters, 3x3 kernel size, & ReLU activation function
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        # Batch normalization to normalize activations 
        BatchNormalization(),
        # Max pooling layer to reduce spatial dimensions of input values
        MaxPooling2D((2, 2)),
        # Dropout layer to prevent overfitting by randomly setting a fraction of input units to 0
        Dropout(0.3),
        
        # 2nd convolutional layer with 64 filters & 3x3 kernel size
        Conv2D(64, (3, 3), activation='relu'),
        # Batch normalization
        BatchNormalization(),
        # Max pooling layer
        MaxPooling2D((2, 2)),
        # Dropout layer
        Dropout(0.3),
        
        # 3rd convolutional layer with 128 filters & 3x3 kernel size
        Conv2D(128, (3, 3), activation='relu'),
        # Batch normalization
        BatchNormalization(),
        # Max pooling layer
        MaxPooling2D((2, 2)),
        # Dropout layer
        Dropout(0.3),
        
        # Flatten layer to convert 3D output to 1D
        Flatten(),
        # Fully connected layer & ReLU activation
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        # Dropout layer
        Dropout(0.6),
        # Output layer with 3 units (for 3 classes) & softmax activation function for multi-class classification
        Dense(3, activation='softmax')
    ])
    # Compile model with Adam optimizer, categorical cross-entropy loss function, & accuracy metric
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to create a CNN model for regression
def create_regression_model(output_dim):
    # Sequential model
    model = Sequential([
        # 1st convolutional layer with 32 filters, 3x3 kernel size, & ReLU activation function
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        # Batch normalization
        BatchNormalization(),
        # Max pooling layer
        MaxPooling2D((2, 2)),
        # Dropout layer
        Dropout(0.3),
        
        # Second convolutional layer with 64 filters & 3x3 kernel size
        Conv2D(64, (3, 3), activation='relu'),
        # Batch normalization
        BatchNormalization(),
        # Max pooling layer
        MaxPooling2D((2, 2)),
        # Dropout layer
        Dropout(0.3),
        
        # 3rd convolutional layer with 128 filters & 3x3 kernel size
        Conv2D(128, (3, 3), activation='relu'),
        # Batch normalization
        BatchNormalization(),
        # Max pooling layer
        MaxPooling2D((2, 2)),
        # Dropout layer
        Dropout(0.3),
        
        # Flatten layer
        Flatten(),
        # Fully connectedlayer & ReLU activation
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        # Dropout layer
        Dropout(0.6),
        # Output layer with output_dim units, & linear activation function for regression
        Dense(output_dim, activation='linear')
        # Constraint to ensure output values are in the range [0, 1]
        #Dense(output_dim, activation='sigmoid')
    ])
    # Compile model with Adam optimizer, mean squared error loss function, & mean absolute error metric
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model
