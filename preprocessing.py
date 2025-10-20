
import pandas as pd
import numpy as np
import argparse
import sys
import warnings
from IPython.display import display

link = 'data/telco_customer_churn.csv'

df = pd.read_csv(link)


def visualization(data):
    df = pd.read_csv(data)
    data_new = df.drop(columns=['customerID'], axis=1)
    data_new['TotalCharges'] = pd.to_numeric(data_new['TotalCharges'], errors='coerce')
    data_new['TotalCharges'] = data_new['TotalCharges'].fillna(data_new['TotalCharges'].median())
    data_c = data_new.copy()
    senior_citizen = {0: 'Young Citizen', 1: 'Senior Citizen'}
    data_c['SeniorCitizen'] = data_c['SeniorCitizen'].map(senior_citizen)
    display(data_c.sample(5))
    return data_c.to_csv('data/data_for_visualization.csv', index=False)


def encode_data(data):
    if data.lower().endswith('.csv'):
        try:
            df = pd.read_csv(data)
            df_c = df.copy()
            df_c['TotalCharges'] = pd.to_numeric(df_c['TotalCharges'], errors='coerce')
            df_c['TotalCharges'] = df_c['TotalCharges'].fillna(df_c['TotalCharges'].median())
            df_c = df_c.drop(columns=['customerID'], axis=1)
            categorical_cols = df_c.select_dtypes(include=['object']).columns.tolist()
            df_encoded = pd.get_dummies(df_c, columns=categorical_cols, drop_first=True, dtype=int)
            return df_encoded.to_csv('data/train_2.csv', index=False)

        except Exception as e:
            warnings.warn(f"Error reading the CSV file: {e}")
    else:
        warnings.warn("The provided file is not a CSV file.")

encode_data(link)