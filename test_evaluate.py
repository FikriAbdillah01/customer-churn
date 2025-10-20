from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import pickle
import pandas as pd

def evaluate_model(model, feature_test, label_test):
    with open(model, 'rb') as f:
        model = pickle.load(f)
    y_pred = model.predict(feature_test)

    print('Confusion Matrix: ')
    print(confusion_matrix(label_test, y_pred))

    print('\nClassification Report: ')
    print(classification_report(label_test, y_pred))

