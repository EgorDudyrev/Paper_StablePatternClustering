from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgba
import pandas as pd
from itertools import cycle


def init_axis(ax, title=None, xticks=None, yticks=None, xticklabels=None, yticklabels=None, grid_params=None):
    if title is not None:
        ax.set_title(title)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)

    if grid_params is not None:
        ax.grid(**grid_params)


def draw_2d_description(
        ax, description: tuple[tuple[float, float, int], tuple[float, float, int]],
        patch_params: dict = None
):
    patch_params = patch_params if patch_params is not None else {}

    x0, x1 = description[0][:2]
    y0, y1 = description[1][:2]
    x0, x1 = max(x0, ax.get_xlim()[0]), min(x1, ax.get_xlim()[1])
    y0, y1 = max(y0, ax.get_ylim()[0]), min(y1, ax.get_ylim()[1])
    ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, **patch_params))


def draw_clustering(ax, cluster_idxs: list[int], clusters_info: pd.DataFrame, cluster_colors: list[str] = None):
    cluster_colors = ['goldenrod', 'navy', 'green', 'red', 'purple'] if cluster_colors is None else cluster_colors
    cluster_colors = cycle(cluster_colors)
    for concept_idx, clr in zip(cluster_idxs, cluster_colors):
        d_stab = clusters_info.at[concept_idx, 'delta_stability']/clusters_info['delta_stability'].max()
        intent = clusters_info.at[concept_idx, 'intent']
        draw_2d_description(
            ax, intent,
            patch_params=dict(
                fc=to_rgba(clr, 0.4), ec=to_rgba(clr, 1*d_stab), linewidth=5, zorder=1,
                label=f'Concept {concept_idx} (∆stab={clusters_info.at[concept_idx, "delta_stability"]})'
            )
        )
