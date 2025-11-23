import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

def load_data(path='house_data.csv'):
    return pd.read_csv(path)

def build_preprocessor(numeric_features):
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numeric_features)
    ])
    return preprocessor

def preprocess(df, fit=True, preprocessor_path='preprocessor.joblib'):
    df = df.copy()
    if 'price' in df.columns:
        X = df.drop(columns=['price'])
        y = df['price']
    else:
        X = df
        y = None
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    preprocessor = build_preprocessor(numeric_features)
    if fit:
        X_trans = preprocessor.fit_transform(X)
        joblib.dump(preprocessor, preprocessor_path)
    else:
        preprocessor = joblib.load(preprocessor_path)
        X_trans = preprocessor.transform(X)
    X_trans = pd.DataFrame(X_trans, columns=numeric_features)
    return X_trans, y

if __name__ == '__main__':
    df = load_data()
    X, y = preprocess(df)
    print('Preprocessing complete. X shape:', X.shape)
