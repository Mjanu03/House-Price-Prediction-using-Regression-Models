import joblib
import pandas as pd
from preprocessing import preprocess, load_data

def predict_example(model_path='best_model.joblib', preprocessor_path='preprocessor.joblib'):
    # load model and preprocessor
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    # example input (replace with real values)
    example = pd.DataFrame([{
        'lot_area': 1200,
        'overall_quality': 7,
        'year_built': 1995,
        'total_rooms': 6
    }])
    # preprocess (using saved preprocessor)
    X_trans = preprocessor.transform(example)
    # ensure columns order - preprocess returns numpy array; model expects same features
    pred = model.predict(X_trans)
    print('Predicted price for example:', pred[0])

if __name__ == '__main__':
    predict_example()
