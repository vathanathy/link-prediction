import networkx as nx
import os
import pandas as pd


class Setup:
    def creation_graph_dataframe():
        """
        Detail:
            The function return the graph of the data training.txt and the dataframe of the train and the test
        Arguments:
            None
        return:
            graph -> nx.Graph()
            df_train -> pd.DataFrame()
            df_test -> pd.DataFrame()
        """

        # Find the location of data
        filename_testing = os.path.join(Setup.path_project(__file__), "data", "testing.txt")
        filename_training = os.path.join(Setup.path_project(__file__), "data", "training.txt")

        # Creation of the graph_train
        graph = nx.Graph()
        digraph = nx.DiGraph()
        with open(filename_training, "r") as f:
            for line in f:
                line = line.split()
                graph.add_nodes_from([line[0], line[1]])
                digraph.add_nodes_from([line[0], line[1]])
                if line[2] == '1':
                    graph.add_edge(line[0], line[1])
                    digraph.add_edge(line[0], line[1])

        # Creation of the dataframes
        df_train = pd.read_csv(filename_training, sep=" ", header=None)
        df_train.columns = ["node_1", "node_2", "output"]

        df_test = pd.read_csv(filename_testing, sep=" ", header=None)
        df_test.columns = ["node_1", "node_2"]

        return graph, digraph, df_train, df_test

    def get_text(id):
        """
        Detail:
            The function returns the full text of a given node
        Arguments:
            id -> integer
        Return:
            fulltext -> string

        """
        # Find the right path of the text folder
        filename = os.path.join(Setup.path_project(__file__), "node_information", "text")

        # Take the txt of the node
        id_text = str(id) + ".txt"
        filename = os.path.join(filename, id_text)

        # Read the text
        f = open(filename, "r")
        fulltext = f.read()

        return fulltext

    def path_project(file):
        """
        Detail:
            It gives the full path of the project ML1_Kaggle
        Arguments:
            file -> string
        Return:
            filename -> string

        """
        filename = file
        while os.path.basename(filename) != "ML1_Kaggle":
            filename = os.path.dirname(filename)
        return filename


if __name__ == "__main__":
    graph, digraph, df_train, df_test = Setup.creation_graph_dataframe()

    print("\n### Graph\n")
    print(nx.info(graph))

    print("\n### Digraph\n")
    print(nx.info(digraph))

    print("\nData Frame train:\n", df_train.head())
    print("\nShape:", df_train.shape)

    print("\nData Frame test:\n", df_test.head())
    print("\nShape:", df_test.shape)

    print("\nText id 300:")
    print(Setup.get_text(300))