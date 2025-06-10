from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix,
    log_loss
)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba= model.predict_proba(X_test)[:, 1]

    print("---Classification Report---")
    print(classification_report(y_test, y_pred))

    print('---Evaluation Metrics---')
    print(f'Accuracy: {accuracy_score(y_test,y_pred)}') #Measures the overall correctness of the predictions
    print(f'Precision_score: {precision_score(y_test,y_pred)}') #Correct positive predictions / all predicted positive
    print(f'Recall: {recall_score(y_test,y_pred)}') #All actual positives
    print(f'F1-Score: {f1_score(y_test,y_pred)}') #Harmonic mean of precision and recall --balance metric
    print(f'ROC AUC: {roc_auc_score(y_test,y_proba)}')
    print(f'Log Loss: {log_loss(y_test,y_proba)}') #Measures probability error, the lower the better

    print('---Evaluation Metrics---')
    print(confusion_matrix(y_test, y_pred))