import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def plot_churn_distribution(data_link):
    data = pd.read_csv(data_link)
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Churn', data=data)
    plt.title('Churn Distribution')
    plt.xlabel('Churn')
    plt.ylabel('Count')
    plt.show()

def plot_feature_distribution(data_link, feature):
    data = pd.read_csv(data_link)
    if data[feature].dtype == 'object':
        fig = plt.figure(figsize=(10, 6))
        sns.countplot(x=data[feature], hue='Churn', data=data)
        plt.title(f'Distribution of {feature} by Churn')
        plt.xlabel(feature)
        plt.ylabel('Count')
        fig.tight_layout()
        plt.show()
    else:
        fig = plt.figure(figsize=(10, 6))
        sns.histplot(data, x=feature, hue='Churn', kde=True, element='step')
        plt.title(f'Distribution of {feature} by Churn')
        plt.xlabel(feature)
        plt.ylabel('Density')
        fig.tight_layout()
        plt.show()

data_link = 'data/telco_customer_churn.csv'