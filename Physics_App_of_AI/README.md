#README.md
# Galaxy Identification Project

## Introduction

This project involves examining and classifying images of galaxies based on their shapes, derived from a crowdsourced dataset.

## Project Structure

The project is organized as follows:

## Running the Project

Recommended to create a virtual environment 
''' 
python3 -m venv myenv                 
source myenv/bin/activate
'''

1. **Install required packages:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Run the example script to train and evaluate models:**
    ```bash
    bash example.sh
    ```

## Tasks and Exercises

### Exercise 0: Data Exploration

- **Objective:** Familiarize yourself with the dataset and create diagnostic plots.
- **Steps:**
  - Load the dataset from `data/images` and `data/labels.csv`.
  - Explore the dataset: visualize sample images and analyze the label distributions.
  - Identify and report any potential issues with the data, such as inconsistencies or missing values.
  - Discuss possible problems with the data recording process, considering the crowdsourced nature of the labels.

### Exercise 1: Classification

- **Objective:** Classify galaxies into three categories: smooth and round with no disk, has a disk, or the image is flawed.
- **Steps:**
  - Use columns `[Class1.1, Class1.2, Class1.3]` from `labels.csv`.
  - Convert probabilities into one-hot encoded labels.
  - Develop a CNN classification model.
  - Train the model and report key performance metrics (e.g., accuracy, precision, recall).
  - Document the process and provide instructions for result reproduction.

### Exercise 2: Regression

- **Objective:** Perform regression tasks on the dataset to answer specific questions about the galaxies.
- **Steps:**
  - Use columns `[Class2.1, Class2.2]` and `[Class7.1, Class7.2, Class7.3]` from `labels.csv`.
  - Develop a CNN regression model to predict the floating-point labels.
  - Train the model and report key performance metrics (e.g., Mean Squared Error, R² score).
  - Document any constraints and how they can be utilized in the model.

### Exercise 3: Regression Continued

- **Objective:** Extend the regression tasks to include more detailed questions about the galaxies.
- **Steps:**
  - Use columns `[Class6.1, Class6.2]` and `[Class8.1, Class8.2, Class8.3, Class8.4, Class8.5, Class8.6, Class8.7]`.
  - Enhance the CNN regression model to handle the additional outputs.
  - Train the model and report performance metrics.
  - Improve the model architecture to correctly classify rare object classes.

### Studies

- **Objective:** Investigate various aspects of model performance and improvements.
- **Studies:**
  - Evaluate if the same architecture (with different output layers) performs well across all tasks.
  - Explore data augmentations to improve classification performance, especially during testing.
  - Investigate if outputs from one task can inform predictions in subsequent tasks.
