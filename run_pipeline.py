from pipeline.load_data import load_data              
from pipeline.preprocess import build_preprocessing_pipeline   
from pipeline.model_train import train_models            
from pipeline.evaluate import evaluate_model           
from sklearn.model_selection import train_test_split  

df = load_data(r"C:\Users\Library\Desktop\ML_CHC\Cervical_cancer_Data\kag_risk_factors_cervical_cancer.csv")                 

features = df.drop("Biopsy", axis=1)                     
labels = df["Biopsy"]

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, stratify=labels, test_size=0.2, random_state=42
)
num_cols = features.select_dtypes(include="number").columns.tolist()
cat_cols = features.select_dtypes(include="object").columns.tolist()

preprocessor = build_preprocessing_pipeline(num_cols, cat_cols)

models = train_models(preprocessor, X_train, y_train, model_dir="models")

for name,model in models.items():
    print(f'\n Evaluating {name}...')
    evaluate_model(model, X_test, y_test)
