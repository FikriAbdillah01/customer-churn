import pandas as pd
from networkx import display


link = 'data/telco_customer_churn.csv'

data = pd.read_csv(link)

# filter string data
str_filtered = data.select_dtypes(include=['object'])
new_data = data.drop(columns='customerID', axis=1)
test_cols = ['gender', 'Partner']
str_test = new_data[test_cols].copy()

# encode with pd.get_dummies
str_test = pd.get_dummies(str_test, dtype=int, drop_first=True)
print(str_test.head())