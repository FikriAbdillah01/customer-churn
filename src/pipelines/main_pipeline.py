import pandas as pd
import numpy as np
import seaborn as sns
import src.pipelines.preprocessing as preprocessing
import argparse
import src.pipelines.visualize as visualize
import src.pipelines.train as train
import src.pipelines.test_evaluate as test_evaluate
from sklearn.model_selection import train_test_split

link = 'data/telco_customer_churn.csv'

if __name__ == "__main__":
    # Preprocess data
    data = preprocessing.encode_data(link)
    df = pd.read_csv('data/train_2.csv')
    X = df.drop(columns=['Churn_Yes'])
    y = df['Churn_Yes']

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

    # Train model
    train.train_logreg(X_train, y_train)
    train.train_decisiontree(X_train, y_train)

    # Evaluate
    print('Evaluation Results for Logistic Regression: ')
    test_evaluate.evaluate_model('model/logreg_bestparam.pkl', X_test, y_test)

    print('Evaluation Results for Decision Tree: ')
    test_evaluate.evaluate_model('model/dectree_bestparam.pkl', X_test, y_test)