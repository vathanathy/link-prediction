from src.setup.setup import Setup
import pandas as pd
import os
import numpy as np

from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import cross_val_score


class OurModel:
    def __init__(self):
        """
        Detail:
            Constructor of the class. It initializes the three models we use (Random Forest, Logistic Regression and
            Gradient Boosting) and the voting classifier. It also stores the important columns used to the model and
            the performance.
        Arguments:
            None
        return:
            None
        """
        self.rf = RandomForestClassifier(n_estimators=20,
                                         max_depth=20,
                                         random_state=0)
        self.mlp = MLPClassifier(activation='logistic',
                                 solver='adam',
                                 alpha=1e-5,
                                 hidden_layer_sizes=(5, 2))
        self.gb = GradientBoostingClassifier(n_estimators=100,
                                             learning_rate=1.0,
                                             max_depth=1,
                                             random_state=0)
        self.voting = VotingClassifier(estimators=[('mlp', self.mlp), ('rf', self.rf), ('gb', self.gb)], voting='hard')

        self.columns = ["Jaccard",
                        "Adamic-Adar",
                        "Preferential Attachment",
                        "Resource Allocation",
                        "Common Neighbors",
                        "Salton Index",
                        "Sorensen Index"]
        self.perf = None

    def fit(self, df_train):
        """
        Detail:
            It trains the model with the training data.
        Arguments:
            df_train -> pd.DataFrame
        return:
            None
        """
        self.voting = self.voting.fit(df_train[self.columns], df_train["output"])

    def predict(self, df_test):
        """
        Detail:
            It predicts the output for a given test data.
        Arguments:
            df_test -> pd.DataFrame
        return:
            None
        """
        return self.voting.predict(df_test[self.columns])

    def performance(self, df_train):
        """
        Detail:
            It evaluates the model by using cross validation.
        Arguments:
            df_train -> pd.DataFrame
        return:
            None
        """
        self.perf = cross_val_score(self.voting, df_train[self.columns], df_train["output"], cv=5, scoring='f1').mean()
        print("F1-score using cross validation: {}%".format(round(self.perf.mean()*100, 2)))


if __name__ == "__main__":
    df_train = pd.read_csv(os.path.join(Setup.path_project(__file__), "data", "data_train_aux.txt"), index_col=0)
    df_test = pd.read_csv(os.path.join(Setup.path_project(__file__), "data", "data_test_aux.txt"), index_col=0)

    print(df_train.head())
    print(df_test.head())

    ourModel = OurModel()
    ourModel.performance(df_train)