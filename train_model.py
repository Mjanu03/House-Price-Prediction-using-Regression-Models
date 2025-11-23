import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from preprocessing import load_data, preprocess

def train(path='house_data.csv', model_out='best_model.joblib'):
    df = load_data(path)
    X, y = preprocess(df, fit=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Baseline: Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    print('Linear Regression -> RMSE:', mean_squared_error(y_test, y_pred_lr, squared=False),
          'MAE:', mean_absolute_error(y_test, y_pred_lr), 'R2:', r2_score(y_test, y_pred_lr))

    # Random Forest with simple grid search
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20]
    }
    grid = GridSearchCV(rf, param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    y_pred = best.predict(X_test)
    print('\nBest RandomForest params:', grid.best_params_)
    print('Random Forest -> RMSE:', mean_squared_error(y_test, y_pred, squared=False),
          'MAE:', mean_absolute_error(y_test, y_pred), 'R2:', r2_score(y_test, y_pred))

    # Save best model
    joblib.dump(best, model_out)
    print(f'Model saved to {model_out}')

if __name__ == '__main__':
    train()
