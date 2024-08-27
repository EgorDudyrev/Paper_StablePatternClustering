from matplotlib.patches import Rectangle


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
