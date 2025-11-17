from networkx import display
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV 
import pickle
import warnings

def train_logreg(feature, label):
    logreg = LogisticRegression(max_iter=5000)
    param = {'C': [0.01, 0.1, 1, 10, 100],
             'solver': ['liblinear', 'lbfgs']}
    grid_search = GridSearchCV(logreg, param, cv=5)
    grid_search.fit(feature, label)
    best_model = grid_search.best_estimator_
    with open('model/logreg_bestparam.pkl', 'wb') as f:
        pickle.dump(best_model, f)

def train_decisiontree(feature, label):
    dtree = DecisionTreeClassifier()
    param = {'max_depth': [3, 5, 7, 9, None],
             'min_samples_split': [2, 5, 10],
             'min_samples_leaf': [1, 2, 4]}
    grid_search = GridSearchCV(dtree, param, cv=5)
    grid_search.fit(feature, label)
    best_model = grid_search.best_estimator_
    with open('model/dtree_bestparam.pkl', 'wb') as f:
        pickle.dump(best_model, f)