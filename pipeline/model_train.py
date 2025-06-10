from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
import os


def train_models(preprocessor, X_train, y_train, model_dir):
    models = {
        'logistic_regression': LogisticRegression(max_iter=500),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    }

    trained_models = {}

    for name, model in models.items():
        print(f'Training {name}...')

        clf = Pipeline([
            ('preprocessing', preprocessor),
            ('classifier', model)
        ])

        clf.fit(X_train, y_train)
        model_path = os.path.join(model_dir, f"{name}.joblib")
        os.makedirs(model_dir, exist_ok=True)
        dump(clf, model_path)
        trained_models[name] = clf

    return trained_models