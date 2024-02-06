# Price Prediction with Kernel Ridge Regression

This code repository contains a Python script for predicting prices using Kernel Ridge Regression. It includes data loading, preprocessing, model selection, and prediction steps. The code is accompanied by two CSV files, "train.csv" and "test.csv," which contain the training and test data, respectively.

## Dependencies
- `numpy` (v1.21.2)
- `pandas` (v1.3.3)
- `matplotlib` (v3.4.3)
- `scikit-learn` (v0.24.2)

Make sure to have these libraries installed in your Python environment before running the script.

## Code Structure

### Data Loading (`data_loading` function)
This function loads the training and test data from the CSV files, preprocesses it, removes NaN values, and interpolates missing data using the k-nearest neighbors imputation technique.

#### Input
- No input arguments.

#### Output
- `X_train`: Matrix of training inputs with features.
- `y_train`: Array of training outputs with labels.
- `X_test`: Matrix of test inputs with features.

### Modeling and Prediction (`modeling_and_prediction` function)
This function defines the Kernel Ridge Regression model, fits it to the training data, and makes predictions on the test data.

#### Input
- `X_train`: Matrix of training inputs with 10 features.
- `y_train`: Array of training outputs.
- `X_test`: Matrix of test inputs with 10 features.

#### Output
- `y_test`: Array of predictions on the test set.

### Main Execution
The main section of the script loads the data, calls the `modeling_and_prediction` function to make predictions, and saves the results in a CSV file named "results.csv" with a single column containing the predicted prices.

## Model Selection
The script includes code (commented out) for model selection using cross-validation. It evaluates different Kernel Ridge Regression models and Gaussian Process Regression models with various kernels and hyperparameters to find the best-performing model.

## Usage
1. Ensure that the required libraries are installed in your Python environment.
2. Place the "train.csv" and "test.csv" files in the same directory as the script.
3. Run the script, which will perform the following steps:
   - Load and preprocess the data.
   - Train the selected model (Kernel Ridge Regression with a polynomial kernel).
   - Make predictions on the test data.
   - Save the predictions in "results.csv."

Once the script has finished running, you will have the predicted prices in "results.csv."

Please note that you can uncomment the model selection code if you want to explore different regression models and their performance.
