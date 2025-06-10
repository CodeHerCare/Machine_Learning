import joblib
import pandas as pd

def load_model(model_dir):
    model_names = ['logistic_regression', 'random_forest', 'xgboost']
    models = {}

    for name, model in model_names:
        model_path = f"{model_dir}/{name}.joblib"
        print(f'Loading {name} model from {model_path}...')
        model[name] = joblib.load(model_path)

    return models

def predict_risk(models, input_df):
    predictions = {}

    for name, model in models.items():
        probas = model.predict_proba(input_df)[:, 1]
        predictions[name] = probas

        return pd.DataFrame(predictions)
