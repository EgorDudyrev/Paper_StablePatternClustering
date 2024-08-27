import pandas as pd
from paspailleur import pattern_structures as PS


def clustering_reward(
    concepts_indices: list[int],
    concepts_info: pd.DataFrame,
    overlap_weight: float,
    n_concepts_weight: float
) -> tuple[float, dict[str, float]]:
    # TODO: Write (or copy-paste) the function based on the code in First_clusterisation_script notebook
    pass


def run_clustering(dataframe: pd.DataFrame, pat_structure: PS.CartesianPS, clustering_params: dict) -> pd.DataFrame:
    """Run the proposed clustering algorithm for the provided data with predefined pattern structure

    Parameters
    ----------
    dataframe: pd.DataFrame
        The original data with numerical, categorical or textual columns
    pat_structure: PS.CartesianPS
        Cartesian Pattern Structure that defines how to treat each column in the data.
        That is, it says what column is numerical, what is categorical, and what is textual.
    clustering_params: dict
        Dictionary with all the clustering parameters.
        For example, weights for overlapping objects when computing the reward value

    Return
    ------
    clusters_df: pd.DataFrame
        DataFrame describing found clusters.
        Every row corresponds to a cluster (in the order in which the clusters were found).
        The columns cover the description of the clusters, their extents,
        and important characteristics like delta-stability.

    """
    # TODO: Write the function based on the code in First_clusterisation_script notebook
    pass
