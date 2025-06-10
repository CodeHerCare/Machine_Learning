import joblib
import pandas as pd

def load_models(model_dir):
    model_names = ['logistic_regression', 'random_forest', 'xgboost']
    models = {}

    for name in model_names:
        model_path = f"{model_dir}/{name}.joblib"
        print(f'Loading {name} model from {model_path}...')
        models[name] = joblib.load(model_path)

    return models

def predict_risk(models, input_df):
    predictions = {}

    for name, model in models.items():
        probas = model.predict_proba(input_df)[:, 1]
        predictions[name] = probas

    return pd.DataFrame(predictions)

if __name__ == "__main__":
    # Load all models from your 'models' directory
    models = load_models("models")

    # Create a dummy input sample (fill with realistic example values or zeros)
    sample_data = {
        'Age': [30],
        'Number of sexual partners': [2],
        'First sexual intercourse': [18],
        'Num of pregnancies': [1],
        'Smokes': [0],
        'Smokes (years)': [0],
        'Smokes (packs/year)': [0],
        'Hormonal Contraceptives': [0],
        'Hormonal Contraceptives (years)': [0],
        'IUD': [0],
        'IUD (years)': [0],
        'STDs': [0],
        'STDs (number)': [0],
        'STDs:condylomatosis': [0],
        'STDs:cervical condylomatosis': [0],
        'STDs:vaginal condylomatosis': [0],
        'STDs:vulvo-perineal condylomatosis': [0],
        'STDs:syphilis': [0],
        'STDs:pelvic inflammatory disease': [0],
        'STDs:genital herpes': [0],
        'STDs:molluscum contagiosum': [0],
        'STDs:AIDS': [0],
        'STDs:HIV': [0],
        'STDs:Hepatitis B': [0],
        'STDs:HPV': [0],
        'STDs: Number of diagnosis': [0],
        'STDs: Time since first diagnosis': [0],
        'STDs: Time since last diagnosis': [0],
        'Dx:Cancer': [0],
        'Dx:CIN': [0],
        'Dx:HPV': [0],
        'Dx': [0],
        'Hinselmann': [0],
        'Schiller': [0],
        'Citology': [0]
    }

    input_df = pd.DataFrame(sample_data)

    # Run predictions on the sample input
    preds_df = predict_risk(models, input_df)
    print("Prediction probabilities for patient risk:")
    print(preds_df)
