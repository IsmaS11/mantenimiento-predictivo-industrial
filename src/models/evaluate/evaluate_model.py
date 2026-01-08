import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


## Función para obtener las métricas de evaluación del modelo
def get_scores(model_name:str, model, X_test_data, y_test_data):
    '''
    Generate a table of test scores.

    In: 
        model_name (string):  How you want your model to be named in the output table
        model:                A fit GridSearchCV object
        X_test_data:          numpy array of X_test data
        y_test_data:          numpy array of y_test data

    Out: pandas df of precision, recall, f1, accuracy, and AUC scores for your model
    '''

    preds = model.best_estimator_.predict(X_test_data)


    precision = precision_score(y_test_data, preds, average='macro')
    recall = recall_score(y_test_data, preds, average='macro')
    f1 = f1_score(y_test_data, preds, average='macro')

    table = pd.DataFrame({'model': [model_name],
                          'precision': [precision], 
                          'recall': [recall],
                          'f1': [f1]
                         })
  
    return table