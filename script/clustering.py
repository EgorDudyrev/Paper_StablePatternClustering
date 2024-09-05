import heapq
from collections import deque
from heapq import nlargest
from typing import Iterable, Union, Iterator

import numpy as np
import pandas as pd
from paspailleur import pattern_structures as PS
import caspailleur as csp

from bitarray import frozenbitarray
from functools import reduce
from itertools import combinations
from bitarray.util import subset as ba_subset, count_and
from tqdm.auto import tqdm


def find_key_dimensions(intent, data, pattern_structure: PS.CartesianPS, top_intent=None) -> tuple[int, ...]:
    """Return (one of) the shortest set of dimension indices from `intent` that describes the same extent"""
    extent = set(pattern_structure.extent(data, intent))
    top_intent = pattern_structure.intent(data) if top_intent is None else top_intent
    non_trivial_dimensions = [i for i, pattern in enumerate(intent) if pattern != top_intent[i]]

    def merge_patterns(dimensions_to_merge: list[int], intent, top_intent):
        new_pattern = list(top_intent)
        for i in dimensions_to_merge:
            new_pattern[i] = intent[i]
        return tuple(new_pattern)

    for level in range(len(non_trivial_dimensions)+1):
        for dimensions_to_change in combinations(non_trivial_dimensions, level):
            merged_pattern = merge_patterns(dimensions_to_change, intent, top_intent)
            next_extent = pattern_structure.extent(data, merged_pattern)
            if all(g in extent for g in next_extent):
                return dimensions_to_change

    raise ValueError('This part of the code should never be accessed')


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


def clustering_reward2(
        concepts_indices: list[int],
        concepts_info: dict[str, list],
        overlap_weight: float = 0,
        n_concepts_weight: float = 0,
        imbalance_weight: float = 0,
        stability_weight: float = 0,
        complexity_weight: float = 0,
        n_concepts_max: int = 10,
) -> tuple[float, dict[str, float]]:
    empty_extent = concepts_info['extent'][0] & ~concepts_info['extent'][0]
    n_objects = len(empty_extent)
    max_support = max(concepts_info['support'])

    if concepts_indices:
        coverage_score = reduce(
            frozenbitarray.__or__, [concepts_info['extent'][idx] for idx in concepts_indices], empty_extent
        ).count()
        coverage_score /= n_objects  # normalisation
    else:
        coverage_score = -np.inf
    coverage_weight = 1

    if len(concepts_indices) > 1 and overlap_weight:
        overlap_score = sum(
            count_and(concepts_info['extent'][idx1], concepts_info['extent'][idx2])
            for idx1, idx2 in combinations(concepts_indices, 2)
        )
        overlap_score /= max_support  # normalisation
    else:
        overlap_score = 0

    n_concepts_score = len(concepts_indices)
    n_concepts_score /= n_concepts_max  # normalisation

    if len(concepts_indices) > 1 and imbalance_weight:
        imbalance_score = np.std([concepts_info['support'][idx] for idx in concepts_indices], ddof=1)
        imbalance_score /= max_support  # normalisation
    else:
        imbalance_score = 0

    if concepts_indices and stability_weight:
        stability_score = sum(concepts_info['delta_stability'][idx] for idx in concepts_indices)/len(concepts_indices)
        max_stability = max(concepts_info['delta_stability'])
        stability_score /= max_stability  # normalisation
    else:
        stability_score = 0

    if concepts_indices and complexity_weight:
        complexity_score = sum(concepts_info['level'][idx] for idx in concepts_indices)/len(concepts_indices)
        complexity_score /= len(concepts_info['intent'][0])  # normalisation by n_dimensions
    else:
        complexity_score = 0

    reward, reward_detailed = 0, {}
    for score_name in ['coverage', 'overlap', 'n_concepts', 'imbalance', 'stability', 'complexity']:
        sign = 1 if score_name in {'coverage', 'stability'} else -1
        reward += locals()[f"{score_name}_weight"] * sign * locals()[f"{score_name}_score"]
        reward_detailed[score_name] = locals()[f"{score_name}_score"]

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


def clusterise_v1(
        concepts_info: dict[str, list],
        overlap_weight: float,
        n_concepts_weight: float,
        imbalance_weight: float,
        stability_weight: float,
        complexity_weight: float,
        thrift_factor: int,
        n_clusters_min: int, n_clusters_max: int,
        return_top_clusterings: bool = False
) -> Union[tuple[list[int], pd.DataFrame], tuple[list[int], pd.DataFrame, pd.DataFrame]]:
    n_concepts = len(concepts_info['extent'])
    reward_params = dict(
        overlap_weight=overlap_weight, n_concepts_weight=n_concepts_weight,
        imbalance_weight=imbalance_weight, stability_weight=stability_weight, complexity_weight=complexity_weight,
        n_concepts_max=n_clusters_max, concepts_info=concepts_info,
    )

    clusterings: dict[tuple[int, ...], float] = {}

    basic_reward = clustering_reward2([], **reward_params)[0]
    queue = deque([([], basic_reward)])
    while queue:
        selected_concepts, selected_reward = queue.popleft()
        if len(selected_concepts) == n_clusters_max:
            clusterings[tuple(selected_concepts)] = selected_reward
            continue

        selected_concepts_set = set(selected_concepts)
        next_rewards = ((next_i, clustering_reward2(selected_concepts+[next_i], **reward_params)[0])
                        for next_i in range(n_concepts) if next_i not in selected_concepts_set)
        next_rewards = {next_i: next_reward for next_i, next_reward in next_rewards if next_reward > selected_reward}

        if not next_rewards and len(selected_concepts) >= n_clusters_min:
            clusterings[tuple(selected_concepts)] = selected_reward

        next_concepts = nlargest(thrift_factor, next_rewards, key=lambda i: next_rewards[i])
        queue.extend([(selected_concepts + [next_i], next_rewards[next_i]) for next_i in next_concepts])

    best_clustering = list(max(clusterings, key=lambda indices: clusterings[indices]))

    rewards_log = pd.DataFrame(
        [clustering_reward2(best_clustering[:i], **reward_params)[1]
         for i in range(len(best_clustering)+1)],
        index=pd.Series(['ø'] + best_clustering, name='Added concept idx')
    )
    best_clusterings_log = pd.DataFrame(
        [{'clustering': indices} | clustering_reward2(indices, **reward_params)[1]
         for indices in heapq.nlargest(5, clusterings, key=lambda indices: clusterings[indices])]
    ).set_index('clustering')

    if return_top_clusterings:
        return best_clustering, rewards_log, best_clusterings_log
    else:
        return best_clustering, rewards_log


def run_clustering(
        dataframe: pd.DataFrame, pat_structure: PS.CartesianPS, clustering_params: dict = None,
        use_tqdm: bool = True
) -> pd.DataFrame:
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
    use_tqdm: bool
        A flag whether to show tqdm progressbar or not

    Return
    ------
    clusters_df: pd.DataFrame
        DataFrame describing found clusters.
        Every row corresponds to a cluster (in the order in which the clusters were found).
        The columns cover the description of the clusters, their extents,
        and important characteristics like delta-stability.

    """
    clustering_params = clustering_params if clustering_params is not None else dict()
    pattern_names = clustering_params.get('column_names')
    min_delta_stability = clustering_params.get('min_delta_stability', 0.01)
    min_support = clustering_params.get('min_support', 0.1)
    max_support = clustering_params.get('max_support', 0.8)
    overlap_weight = clustering_params.get('overlap_weight', 0.01)
    n_concepts_weight = clustering_params.get('n_concepts_weight', 0.01)

    data = list(pat_structure.preprocess_data(dataframe))
    assert len(list(pat_structure.extent(data, pat_structure.intent(data)))) == len(data)
    attributes, attr_extents = zip(*pat_structure.iter_attributes(data, min_support))

    stable_extents = csp.mine_equivalence_classes.list_stable_extents_via_gsofia(
        attr_extents,
        n_objects=len(data), min_delta_stability=min_delta_stability, min_supp=min_support,
        use_tqdm=use_tqdm, n_attributes=len(attr_extents)
    )
    concepts_df = pd.DataFrame(mine_clusters_info(attr_extents, data, max_support, pat_structure, stable_extents, pattern_names, use_tqdm))

    clustering, reward_log = clusterise_v0(concepts_df, overlap_weight, n_concepts_weight)
    clusters_df = concepts_df.loc[clustering]

    return clusters_df


def mine_clusters_info(
        stable_extents: Iterable[frozenbitarray],
        attr_extents: list[frozenbitarray], pat_structure: PS.CartesianPS,
        data: list,
        min_support: float, max_support: float,
        pattern_names: list[str] = None, use_tqdm: bool = False
) -> dict[str, list]:
    pattern_names = [f"x{i}" for i in range(len(pat_structure.basic_structures))]

    stable_extents = [ext for ext in stable_extents if min_support*len(ext) <= ext.count() <= max_support*len(ext)]
    stable_extents = sorted(stable_extents, key=lambda ext: ext.count(), reverse=True)

    stable_intents = [pat_structure.intent(data, ext.search(True))
                      for ext in tqdm(stable_extents, desc='Compute intents', disable=not use_tqdm)]

    delta_stabilities = [csp.indices.delta_stability_by_description(
        describe_with_attributes(extent, attr_extents), attr_extents)
        for extent in stable_extents
    ]

    top_intent = pat_structure.intent(data)
    levels = [len(find_key_dimensions(intent, data, pat_structure, top_intent)) for intent in stable_intents]

    return dict(
        extent=stable_extents,
        intent=stable_intents,
        delta_stability=delta_stabilities,
        support=list(map(frozenbitarray.count, stable_extents)),
        frequency=list(map(lambda extent: extent.count() / len(extent), stable_extents)),
        intent_human=list(map(lambda intent: pat_structure.verbalize(intent, pattern_names=pattern_names), stable_intents)),
        level=levels

    )


def mine_rare_itemsets(
        attribute_extents: list[frozenbitarray], max_support: int,
        max_length: int = 5
) -> Iterator[tuple[frozenbitarray, ...]]:
    from icecream import ic

    n_attrs = len(attribute_extents)
    total_extent = attribute_extents[0] | ~attribute_extents[0]

    prev_level_generators, cur_level_gens = None, {tuple(): total_extent.count()}
    for level in range(1, max_length+1):
        prev_level_generators, cur_level_gens = cur_level_gens, {}
        if not prev_level_generators:
            break
        # ic(level)
        # ic(prev_level_generators)

        for old_generator in prev_level_generators:
            old_extent = reduce(frozenbitarray.__and__, map(attribute_extents.__getitem__, old_generator), total_extent)
            next_attr_start = old_generator[-1]+1 if old_generator else 0
            for next_attr in range(next_attr_start, n_attrs):
                new_generator = old_generator + (next_attr,)
                new_support = (old_extent & attribute_extents[next_attr]).count()
                sub_generators = (new_generator[:i] + new_generator[i+1:] for i in range(level))
                # TODO: Optimise for when given a maximal pairwise overlap
                not_a_generator = any(
                    sub_gen not in prev_level_generators or new_support == prev_level_generators[sub_gen]
                    for sub_gen in sub_generators
                )
                # ic(old_generator, new_generator, next_attr, not_a_generator)
                if not_a_generator:
                    continue

                if new_support < max_support:
                    yield new_generator
                    continue

                cur_level_gens[new_generator] = new_support
