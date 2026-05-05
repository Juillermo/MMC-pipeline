# From Michele Coscia (Coscia M, Neffke FMH. Network Backboning with Noisy Data. in 2017 IEEE 33rd International Conference on Data Engineering (ICDE) 425–436 (IEEE, San Diego, CA, USA, 2017). doi:10.1109/ICDE.2017.100.)
import pandas as pd
from scipy.stats import binom


def noise_corrected(table, undirected=False, return_self_loops=False, calculate_p_value=False):
    # print("Calculating NC score...")
    table = table.copy()
    src_sum = table.groupby(by="src").sum()[["nij"]]
    table = table.merge(src_sum, left_on="src", right_index=True, suffixes=("", "_src_sum"))
    trg_sum = table.groupby(by="trg").sum()[["nij"]]
    table = table.merge(trg_sum, left_on="trg", right_index=True, suffixes=("", "_trg_sum"))
    table.rename(columns={"nij_src_sum": "ni.", "nij_trg_sum": "n.j"}, inplace=True)
    table["n.."] = table["nij"].sum()
    table["mean_prior_probability"] = ((table["ni."] * table["n.j"]) / table["n.."]) * (1 / table["n.."])

    if calculate_p_value:
        table["score"] = binom.cdf(table["nij"], table["n.."], table["mean_prior_probability"])
    else:
        table["kappa"] = table["n.."] / (table["ni."] * table["n.j"])
        table["score"] = ((table["kappa"] * table["nij"]) - 1) / ((table["kappa"] * table["nij"]) + 1)
        table["var_prior_probability"] = (1 / (table["n.."] ** 2)) * (
                table["ni."] * table["n.j"] * (table["n.."] - table["ni."]) * (table["n.."] - table["n.j"])) / (
                                                 (table["n.."] ** 2) * ((table["n.."] - 1)))
        table["alpha_prior"] = (((table["mean_prior_probability"] ** 2) / table["var_prior_probability"]) * (
                1 - table["mean_prior_probability"])) - table["mean_prior_probability"]
        table["beta_prior"] = (table["mean_prior_probability"] / table["var_prior_probability"]) * (
                1 - (table["mean_prior_probability"] ** 2)) - (1 - table["mean_prior_probability"])
        table["alpha_post"] = table["alpha_prior"] + table["nij"]
        table["beta_post"] = table["n.."] - table["nij"] + table["beta_prior"]
        table["expected_pij"] = table["alpha_post"] / (table["alpha_post"] + table["beta_post"])
        table["variance_nij"] = table["expected_pij"] * (1 - table["expected_pij"]) * table["n.."]
        table["d"] = (1.0 / (table["ni."] * table["n.j"])) - (
                table["n.."] * ((table["ni."] + table["n.j"]) / ((table["ni."] * table["n.j"]) ** 2)))
        table["variance_cij"] = table["variance_nij"] * (((2 * (table["kappa"] + (table["nij"] * table["d"]))) / (
                ((table["kappa"] * table["nij"]) + 1) ** 2)) ** 2)
        table["sdev_cij"] = table["variance_cij"] ** .5

    if not return_self_loops:
        table = table[table["src"] != table["trg"]]
    if undirected:
        table = table[table["src"] <= table["trg"]]

    return table  # [["src", "trg", "nij", "score"]] if calculate_p_value else table[["src", "trg", "nij", "score", "sdev_cij"]]


def thresholding(table, threshold):
    """Reads a preprocessed edge table and returns only the edges supassing a significance threshold.

    Args:
    table (pandas.DataFrame): The edge table.
    threshold (float): The minimum significance to include the edge in the backbone.

    Returns:
    The network backbone.
    """
    table = table.copy()
    if "sdev_cij" in table:
        return table[(table["score"] - (threshold * table["sdev_cij"])) > 0]  # [["src", "trg", "nij", "score"]]
    else:
        return table[table["score"] > threshold]  # [["src", "trg", "nij", "score"]]


def perform_backboning(data, thresh, sig, calculate_p_value=False):
    assert thresh is not None

    table = data.copy()  # ["i", "j", "ABC", "a_sig", "namei", "namej"]].copy()
    table.rename(columns={"ABC": "nij", "i": "src", "j": "trg"}, inplace=True)
    if sig:
        table = table[table["a_sig"] == True]
    table2 = table.copy()
    table2["new_src"] = table["trg"]
    table2["new_trg"] = table["src"]
    table2.drop("src", 1, inplace=True)
    table2.drop("trg", 1, inplace=True)
    table2 = table2.rename(columns={"new_src": "src", "new_trg": "trg"})
    table = pd.concat([table, table2], axis=0)
    table = table.drop_duplicates(subset=["src", "trg"])
    original_nodes = len(set(table["src"]) | set(table["trg"]))
    nnodes = len(set(table["src"]) | set(table["trg"]))
    nnedges = table.shape[0] / 2

    # if method == "noise_corrected":
    nc_table = noise_corrected(table, undirected=True, calculate_p_value=calculate_p_value)
    nc_backbone = thresholding(nc_table, threshold=thresh)
    # elif method in (
    #         "doubly_stochastic", "disparity_filter", "high_salience_skeleton", "naive", "maximum_spanning_tree"):
    #     nc_backbone = backboning.__dict__[method](table)
    # else:
    #     raise Exception(f"Filtering method '{method}' not understood.")

    return nc_backbone.rename(columns={"nij": "ABC", "src": "i", "trg": "j"})