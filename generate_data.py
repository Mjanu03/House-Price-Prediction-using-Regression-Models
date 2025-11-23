import pandas as pd
import numpy as np
from sklearn.datasets import make_regression

def generate_synthetic_house_data(n_samples=1000, n_features=8, noise=0.1, random_state=42, out_path='house_data.csv'):
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=random_state)
    cols = [f'feature_{i+1}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    # create some interpretable-like features
    df['lot_area'] = (np.abs(df['feature_1']) * 500) + 500  # approx lot size
    df['overall_quality'] = (np.clip((df['feature_2'] * 3 + 5).round().astype(int), 1, 10))
    df['year_built'] = (np.clip((2000 + (df['feature_3'] * 30)).round().astype(int), 1900, 2025))
    df['total_rooms'] = (np.clip((df['feature_4'] * 3 + 5).round().astype(int), 2, 12))
    # drop raw features to keep a few meaningful ones
    df = df[['lot_area', 'overall_quality', 'year_built', 'total_rooms']]
    # construct price (target) using a combination + noise
    price = (df['lot_area'] * 50) + (df['overall_quality'] * 10000) + ((df['year_built'] - 1900) * 100) + (df['total_rooms'] * 2000)
    rng = np.random.RandomState(random_state)
    price = price + rng.normal(0, 20000, size=len(price))
    df['price'] = price.round(2)
    df.to_csv(out_path, index=False)
    print(f'Synthetic house price data saved to {out_path}')

if __name__ == '__main__':
    generate_synthetic_house_data()
