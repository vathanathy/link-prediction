from src.setup.setup import Setup
from src.features.GraphTopology import GraphTopology
from src.models.OurModel import OurModel
import os
import pandas as pd

"""
Our program occurs in this file
"""

if __name__=="__main__":
    # Taking the graph and the teste of training
    print("\nCreating the graph and digraph...\n")
    graph, digraph, df_train, df_test = Setup.creation_graph_dataframe()

    # Feature Engineering
    print("\nFeature Engineering...\n")
    df_train, df_test = GraphTopology.fit(graph, digraph, df_train, df_test)

    # Modelling and Evaluation
    ourModel = OurModel()
    print("\nCalculating performance...\n")
    ourModel.performance(df_train)
    print("\nFitting the model...\n")
    ourModel.fit(df_train)

    # Prediction
    print("\nPredicting the data test...\n")
    pred = ourModel.predict(df_test)
    pred = pd.DataFrame(pred)
    pred.columns = ["predicted"]
    pred.index.name = "id"

    # CSV
    pred.to_csv(os.path.join(Setup.path_project(__file__), "data", "predictions.csv"))
