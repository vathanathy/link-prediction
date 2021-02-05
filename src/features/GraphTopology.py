import networkx as nx
import numpy as np
import pandas as pd
import os
from src.setup.setup import Setup


class GraphTopology:
    def fit(graph, digraph, df_train, df_test):
        """
        Detail:
            It adds the metric coefficients and the neighbor coefficients as columns in the dataset
        Arguments:
            graph -> nx.Graph()
            df_train -> pd.DataFrame()
            df_test -> pd.DataFrame()
        Return:
            df_train -> pd.DataFrame()
            df_test -> pd.DataFrame()

        """
        print("\n\tmetric_coefficients...\n")
        df_train, df_test = GraphTopology.metric_coefficients(graph, df_train, df_test)
        print("\n\tneighborhood_coefficients...\n")
        df_train, df_test = GraphTopology.neighborhood_coefficients(graph, df_train, df_test)
        print("\n\tcoefficients_digraph...\n")
        df_train, df_test = GraphTopology.coefficients_digraph(digraph, df_train, df_test)
        return df_train, df_test

    def metric_coefficients(graph, df_train, df_test):
        """
        Detail:
            It computes the metric coefficients like jaccard, adamic,preferential attachment and resource allocation
        Arguments:
            graph -> nx.Graph()
            df_train -> pd.DataFrame()
            df_test -> pd.DataFrame()
        Return:
            df_train -> pd.DataFrame()
            df_test -> pd.DataFrame()

        """
        filename_testing = os.path.join(Setup.path_project(__file__), "data", "testing.txt")
        filename_training = os.path.join(Setup.path_project(__file__), "data", "training.txt")

        for filename, df in zip([filename_training, filename_testing], [df_train, df_test]):
            jaccard = []
            adamic_adar = []  # Adamic-Adar inde
            pa = []  # preferential attachment
            ra = []  # resource allocation

            with open(filename, "r") as f:
                for line in f:
                    line = line.split()
                    for u, v, p in nx.jaccard_coefficient(graph, [(line[0], line[1])]):
                        jaccard.append(p)
                    for u, v, p in nx.adamic_adar_index(graph, [(line[0], line[1])]):
                        adamic_adar.append(p)
                    for u, v, p in nx.preferential_attachment(graph, [(line[0], line[1])]):
                        pa.append(p)
                    for u, v, p in nx.resource_allocation_index(graph, [(line[0], line[1])]):
                        ra.append(p)

            df["Jaccard"] = jaccard
            df["Adamic-Adar"] = adamic_adar
            df["Preferential Attachment"] = pa
            df["Resource Allocation"] = ra

        return df_train, df_test

    def neighborhood_coefficients(graph, df_train, df_test):
        """
        Detail:
            It computes the neighborhood coefficients like common neighbors, salton index and sorensen index
        Arguments:
            graph -> nx.Graph()
            df_train -> pd.DataFrame()
            df_test -> pd.DataFrame()
        Return:
            df_train -> pd.DataFrame()
            df_test -> pd.DataFrame()

        """
        def intersection(lst1, lst2):
            return list(set(lst1) & set(lst2))

        filename_testing = os.path.join(Setup.path_project(__file__), "data", "testing.txt")
        filename_training = os.path.join(Setup.path_project(__file__), "data", "training.txt")

        for filename, df in zip([filename_training, filename_testing], [df_train, df_test]):
            cn = []  # common neighbors
            si = []  # salton index
            sorI = []  # sorensen index
            hpi = []  # Hub Promoted Index
            hdi = []  # Hub Depressed Index
            lhni = []  # Leicht–Holme–Newman Index

            with open(filename, "r") as f:
                for line in f:
                    line = line.split()

                    n1 = graph.neighbors(line[0])
                    n2 = graph.neighbors(line[1])
                    inter = len(intersection(n1, n2))

                    cn.append(inter)
                    if graph.degree(line[0]) != 0 and graph.degree(line[1]) != 0:
                        si.append(inter / np.sqrt(graph.degree(line[0]) * graph.degree(line[1])))
                    else:
                        si.append(0)
                    sorI.append(2 * inter / (graph.degree(line[0]) + graph.degree(line[1])))

                    if graph.degree(line[0]) != 0 and graph.degree(line[1]) != 0:
                        hpi.append(inter / np.minimum(graph.degree(line[0]), graph.degree(line[1])))
                        hdi.append(inter / np.maximum(graph.degree(line[0]), graph.degree(line[1])))
                        lhni.append(inter / graph.degree(line[0]) * graph.degree(line[1]))
                    else:
                        hpi.append(0)
                        hdi.append(0)
                        lhni.append(0)

            df["Common Neighbors"] = cn
            df["Salton Index"] = si
            df["Sorensen Index"] = sorI
            df["Hub Promoted Index"] = hpi
            df["Hub Depressed Index"] = hdi
            df["Leicht–Holme–Newman Index"] = lhni

        return df_train, df_test

    def coefficients_digraph(digraph, df_train, df_test):
        def intersection(lst1, lst2):
            return set(set(lst1) & set(lst2))

        filename_testing = os.path.join(Setup.path_project(__file__), "data", "testing.txt")
        filename_training = os.path.join(Setup.path_project(__file__), "data", "training.txt")

        for filename, df in zip([filename_training, filename_testing], [df_train, df_test]):
            disp = []  # dispersion
            lh_A = []  # likelihood given A
            lh_D = []  # likelihood given D

            ded = []  # deductive metric
            ind = []  # inductive metric
            inf = []  # inference score
            inf_2d = []  # modified inference

            ded_log = []
            ind_log = []
            inf_log = []
            inf_log_2d = []

            # ded_2 = []
            # ind_2 = []
            # inf_ql = []

            abd = []

            with open(filename, "r") as f:
                for line in f:
                    line = line.split()

                    # D is the descendant and A is the ancestor of a node.
                    # In this case, A1 is the ancestor of the line[0]
                    Dx = set(digraph.predecessors(line[0]))
                    Ax = set(digraph.successors(line[0]))
                    Dy = set(digraph.predecessors(line[1]))
                    Ay = set(digraph.successors(line[1]))

                    # Dx_2 = None
                    # Ax_2 = None
                    # Dy_2 = None
                    # Ay_2 = None
                    # alpha = None

                    AYx = intersection(Ax, Dy)
                    DYx = intersection(Dx, Dy)

                    disp.append(nx.dispersion(digraph, line[0], line[1]))

                    if len(Ax) == 0:
                        lh_A.append(0)
                        ded.append(0)
                        ded_log.append(0)
                        abd.append(0)
                    else:
                        lh_A.append(len(AYx)/len(Ax))
                        ded.append(len(intersection(Ax, Dy))/len(Ax))
                        ded_log.append(len(intersection(Ax, Dy))/len(Ax) * np.log(len(Ax)))
                        abd.append(len(intersection(Ax, Ay))/len(Ax))

                    if len(Dx) == 0:
                        lh_D.append(0)
                        ind.append(0)
                        ind_log.append(0)
                    else:
                        lh_D.append(len(DYx)/len(Dx))
                        ind.append(len(intersection(Dx, Dy)) / len(Dx))
                        ind_log.append(len(intersection(Dx, Dy)) / len(Dx) * np.log(len(Dx)))

                    # if len(Ax_2) == 0:
                    #     ded_2.append(ded[-1])
                    # else:
                    #     ded_2.append( len(intersection(Ax_2, Dy_2))/(len(Ax_2) * alpha) + ded[-1])
                    #
                    # if len(Dx_2) == 0:
                    #     ind_2.append(ind[-1])
                    # else:
                    #     ind_2.append( len(intersection(Dx_2, Dy_2))/(len(Dx_2) * alpha) + ind[-1])

                    inf.append(ded[-1] + ind[-1])
                    inf_2d.append(2*ded[-1] + ind[-1])
                    inf_log.append(ded_log[-1] + ind_log[-1])
                    inf_log_2d.append(2 * ded_log[-1] + ind_log[-1])
                    # inf_ql.append(ded_2[-1] + ind_2[-1])

            df["Dispersion"] = disp

            df["Likelihood A"] = lh_A
            df["Likelihood D"] = lh_D

            df["Deductive"] = ded
            df["Inductive"] = ind
            df["Inference"] = inf
            df["Inference 2D"] = inf_2d

            df["Deductive log"] = ded_log
            df["Inductive log"] = ind_log
            df["Inference log"] = inf_log
            df["Inference log 2D"] = inf_log_2d

            # df["Deductive square"] = ded_2
            # df["Inductive square"] = ind_2
            # df["Inference QL"] = inf_ql

            df["Abductive"] = abd

        return df_train, df_test


if __name__ == "__main__":
    graph, digraph, df_train, df_test = Setup.creation_graph_dataframe()

    print("### Data train:\n")
    print(df_train.head())
    print("\n### Data test:\n")
    print(df_test.head())

    df_train, df_test = GraphTopology.metric_coefficients(graph, df_train, df_test)
    print("\n####### After metric coefficients:\n")
    print("### Data train:\n")
    print(df_train.head())
    print("\n### Data test:\n")
    print(df_test.head())

    df_train, df_test = GraphTopology.neighborhood_coefficients(graph, df_train, df_test)
    print("\n####### After neighborhood coefficients:\n")
    print("### Data train:\n")
    print(df_train.head())
    print("\n### Data test:\n")
    print(df_test.head())

    df_train, df_test = GraphTopology.coefficients_digraph(digraph, df_train, df_test)
    print("\n####### After coefficients digraph:\n")
    print("### Data train:\n")
    print(df_train.head())
    print("\n### Data test:\n")
    print(df_test.head())

    print(df_train.describe())

    df_train.to_csv(os.path.join(Setup.path_project(__file__), "data", "data_train_aux.txt"))
    df_test.to_csv(os.path.join(Setup.path_project(__file__), "data", "data_test_aux.txt"))
