import pandas as pd
from paspailleur import pattern_structures as PS
import caspailleur as csp

from bitarray import frozenbitarray
from functools import reduce
from itertools import combinations
from bitarray.util import subset as ba_subset
from tqdm.auto import tqdm


def clustering_reward(
    concepts_indices: list[int], 
    concepts_info: pd.DataFrame, 
    overlap_weight: float, 
    n_concepts_weight: float
) -> tuple[float, dict[str, float]]:
    empty_extent = concepts_info['extent'].iat[0] & ~concepts_info['extent'].iat[0]

    overlaps_per_pairs_of_concepts = [concepts_info.at[idx1, 'extent'] & concepts_info.at[idx2, 'extent']
                                      for idx1, idx2 in combinations(concepts_indices, 2)]
    overlapping_objects = reduce(frozenbitarray.__or__, overlaps_per_pairs_of_concepts, empty_extent)
    covered_objects = reduce(frozenbitarray.__or__, [concepts_info.at[idx, 'extent'] for idx in concepts_indices], empty_extent)

    reward_detailed = {
        'total_cover': covered_objects.count()/len(covered_objects),
        'overlap': overlapping_objects.count()/len(overlapping_objects),
        'n_concepts': len(concepts_indices)
    }
    
    reward = reward_detailed['total_cover'] - overlap_weight * reward_detailed['overlap'] - n_concepts_weight * reward_detailed['n_concepts']
    reward_detailed['reward'] = reward
    return reward, reward_detailed


def describe_with_attributes(extent, attr_extents) -> list[int]:
    """Find attribute-based intent for a given extent. So not a 'pattern intent'"""
    return [i for i, attr_extent in enumerate(attr_extents) if ba_subset(extent, attr_extent)]


def clusterise_v0(
    concepts_info: pd.DataFrame, 
    overlap_weight: float, 
    n_concepts_weight: float
) -> tuple[list[int], pd.DataFrame]:
    selected_concepts = []
    reward, reward_detailed = clustering_reward([], concepts_info, overlap_weight, n_concepts_weight)
    rewards_log = [reward_detailed]

    while True:
        old_reward = reward
        for next_cluster_candidate in concepts_info.index:
            next_reward, next_detailed = clustering_reward(
                selected_concepts+[next_cluster_candidate], concepts_info, overlap_weight, n_concepts_weight
            )
            if next_reward > reward:
                reward, reward_detailed = next_reward, next_detailed
                next_cluster_idx = next_cluster_candidate
        
        if reward == old_reward:
            break
        
        selected_concepts.append(next_cluster_idx)
        rewards_log.append(reward_detailed)

    rewards_log = pd.DataFrame(rewards_log, index=pd.Series(['ø']+selected_concepts, name='Added concept idx'))
    return selected_concepts, rewards_log


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

    pattern_names = clustering_params.get('column_names', ['x0','x1'])
    min_delta_stability = clustering_params.get('min_delta_stability', 0.01)
    min_support = clustering_params.get('min_support', 0.1)
    max_support = clustering_params.get('max_support', 0.8)
    overlap_weight = clustering_params.get('overlap_weight', 0.01)
    n_concepts_weight = clustering_params.get('n_concepts_weight', 0.01)

    data = list(pat_structure.preprocess_data(dataframe))
    assert len(list(pat_structure.extent(data, pat_structure.intent(data)))) == len(data)
    attributes, attr_extents = zip(*pat_structure.iter_attributes(data, min_support))

    stable_extents = csp.mine_equivalence_classes.list_stable_extents_via_gsofia(
        attr_extents, len(data), min_delta_stability, min_support, use_tqdm=True, n_attributes=len(attr_extents)
    )
    stable_extents = sorted(stable_extents, key=lambda ext: ext.count(), reverse=True)
    stable_intents = [pat_structure.intent(data, ext.search(True)) for ext in tqdm(stable_extents)]

    delta_stabilities = [
        csp.indices.delta_stability_by_description(
            describe_with_attributes(extent, attr_extents), attr_extents)
        for extent in stable_extents
    ]

    concepts_df = pd.DataFrame(dict(
        extent=stable_extents,
        intent=stable_intents,
        delta_stability=delta_stabilities,
        support=map(frozenbitarray.count, stable_extents),
        frequency=map(lambda extent: extent.count()/len(extent), stable_extents),
        intent_human=map(lambda intent: pat_structure.verbalize(intent, pattern_names), stable_intents)
    ))

    concepts_df = concepts_df[concepts_df['frequency'] < max_support]

    clustering, reward_log = clusterise_v0(concepts_df, clustering_params[overlap_weight], clustering_params[n_concepts_weight])

    clusters_df = concepts_df.loc[clustering]

    return clusters_df 
