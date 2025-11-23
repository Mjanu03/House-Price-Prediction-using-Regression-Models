# House Price Prediction - Regression Models (Sample Project)

This repository contains a sample house price prediction project using Python. It demonstrates an end-to-end workflow:
synthetic data generation, preprocessing, exploratory data analysis (EDA), model training and prediction.

## Files
- `generate_data.py` : Generates a synthetic house price dataset and saves it as `house_data.csv`.
- `preprocessing.py` : Loads data, imputes missing values, scales numeric features, and saves the preprocessor.
- `eda.py` : Performs basic exploratory data analysis and saves plots/csv in `eda_outputs/`.
- `train_model.py` : Trains Linear Regression and a RandomForest regressor with simple grid search, evaluates metrics, and saves the best model.
- `predict.py` : Example script to load the saved model and make a prediction.
- `requirements.txt` : Python libraries required.

## How to run
1. Create a virtual environment and install requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\\Scripts\\activate
   pip install -r requirements.txt
   ```
2. Generate data:
   ```bash
   python generate_data.py
   ```
3. Run EDA:
   ```bash
   python eda.py
   ```
4. Preprocess and train:
   ```bash
   python train_model.py
   ```
5. Predict example:
   ```bash
   python predict.py
   ```

## Notes
- Replace the synthetic data with real datasets as needed.
- Tune hyperparameters and add cross-validation for production use.
