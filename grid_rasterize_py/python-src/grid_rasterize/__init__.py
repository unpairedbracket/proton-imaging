import timeit
import numpy as np
from numpy import floor, ceil

from numba import jit

from . import rust_bindings


def get_intensity(
    x0, y0, dx, dy, pixel_zoom, xlim, ylim, background_intensity=None, level=0
):
    x0 = x0 * pixel_zoom
    y0 = y0 * pixel_zoom
    dx = dx * pixel_zoom
    dy = dy * pixel_zoom

    if xlim:
        xlim = tuple(l * pixel_zoom for l in xlim)
    if ylim:
        ylim = tuple(l * pixel_zoom for l in ylim)

    edg, ext, secs = rasterize_grid(
        x0, y0, dx, dy, xlim, ylim, background_intensity, level
    )

    print("Summing edge values...")
    I = np.cumsum(edg, axis=0)
    ext = tuple(l / pixel_zoom for l in ext)
    return I, ext, secs


def rasterize_grid(
    x0, y0, dx, dy, xlim=None, ylim=None, background_intensity=None, level=0
):
    X = x0 + dx
    Y = y0 + dy

    orig_areas = (
        np.diff(x0[:, :-1] + x0[:, 1:], axis=0)
        * np.diff(y0[:-1, :] + y0[1:, :], axis=1)
        / 8
    )

    if background_intensity:
        X_c = (X[1:, 1:] + X[1:, :-1] + X[:-1, 1:] + X[:-1, :-1]) / 4
        Y_c = (Y[1:, 1:] + Y[1:, :-1] + Y[:-1, 1:] + Y[:-1, :-1]) / 4
        orig_areas *= background_intensity(X_c, Y_c)

    print("Generating Edges...")
    edges, degenerate_tris = get_edges(X, Y, orig_areas)

    print("Clipping for contribution...")
    edges = clip_contribution(edges)

    if ylim:
        Y_min = int(ceil(ylim[0]) - 1)
        Y_max = int(floor(ylim[1]) + 1)
        print("Clipping to y bounds...")
        edges = clip_to_y(edges, ylim)
    else:
        Y_min = int(ceil(Y.min()) - 1)
        Y_max = int(floor(Y.max()) + 1)

    if xlim:
        X_min = int(ceil(xlim[0]) - 1)
        X_max = int(floor(xlim[1]) + 1)
        print("Clipping to x bounds...")
        pre_edges, edges = clip_to_x(edges, xlim)
    else:
        X_min = int(ceil(X.min()) - 1)
        X_max = int(floor(X.max()) + 1)
        pre_edges = np.array([])

    raster = np.zeros((X_max - X_min + 1, Y_max - Y_min))

    print("Initialising left edge...")
    for edge in pre_edges:
        fill_starting(raster[0, :], edge, Y_min)

    print("Filling accumulation buffer...")

    seconds = 0
    if level == 0:
        seconds = timeit.timeit(
            "fill_accumulation_py(raster, edges, (X_min, Y_min))",
            number=1,
            globals=globals() | locals(),
        )
    elif level == 1:
        seconds = timeit.timeit(
            "fill_accumulation_numba(raster, edges, (X_min, Y_min))",
            number=1,
            globals=globals() | locals(),
        )
    elif level == 2:
        seconds = timeit.timeit(
            "rust_bindings.fill_accumulation_rs(raster, edges, (X_min, Y_min))",
            number=1,
            globals=globals() | locals(),
        )

    for tri in degenerate_tris:
        add_degenerate(raster, tri)

    return raster, (X_min, X_max + 1, Y_min, Y_max), seconds


def fill_accumulation_numba(raster, edges, xy_min):
    for edge in edges:
        fill_accumulation(raster, edge, xy_min)


def fill_accumulation_py(raster, edges, xy_min):
    for edge in edges:
        fill_accumulation_nojit(raster, edge, xy_min)


def get_edges(X, Y, flat_areas):
    quad_shape = X.shape
    X_c = (X[1:, 1:] + X[1:, :-1] + X[:-1, 1:] + X[:-1, :-1]) / 4
    Y_c = (Y[1:, 1:] + Y[1:, :-1] + Y[:-1, 1:] + Y[:-1, :-1]) / 4

    # Edge format: x_0, y_0, x_1, y_1, weight
    x_edges = np.zeros((quad_shape[0], quad_shape[1] - 1, 5))
    y_edges = np.zeros((quad_shape[0] - 1, quad_shape[1], 5))

    # Internal edge order (dim 2): NW, NE, SE, SW
    internal_edges = np.zeros((quad_shape[0] - 1, quad_shape[1] - 1, 4, 5))

    x_edges[:, :, 0] = X[:, :-1]
    x_edges[:, :, 1] = Y[:, :-1]
    x_edges[:, :, 2] = X[:, 1:]
    x_edges[:, :, 3] = Y[:, 1:]

    y_edges[:, :, 0] = X[:-1, :]
    y_edges[:, :, 1] = Y[:-1, :]
    y_edges[:, :, 2] = X[1:, :]
    y_edges[:, :, 3] = Y[1:, :]

    internal_edges[:, :, 0, 0] = X[:-1, 1:]
    internal_edges[:, :, 1, 0] = X[1:, 1:]
    internal_edges[:, :, 2, 0] = X[1:, :-1]
    internal_edges[:, :, 3, 0] = X[:-1, :-1]

    internal_edges[:, :, 0, 1] = Y[:-1, 1:]
    internal_edges[:, :, 1, 1] = Y[1:, 1:]
    internal_edges[:, :, 2, 1] = Y[1:, :-1]
    internal_edges[:, :, 3, 1] = Y[:-1, :-1]

    internal_edges[:, :, :, 2] = X_c[:, :, None]
    internal_edges[:, :, :, 3] = Y_c[:, :, None]

    # Area order N, E, S, W; actually 2*area stored
    areas = np.zeros((quad_shape[0] - 1, quad_shape[1] - 1, 4))

    areas[:, :, 0] = (Y[1:, 1:] - Y_c) * (X_c - X[:-1, 1:]) - (X[1:, 1:] - X_c) * (
        Y_c - Y[:-1, 1:]
    )
    areas[:, :, 1] = (Y[1:, :-1] - Y_c) * (X_c - X[1:, 1:]) - (X[1:, :-1] - X_c) * (
        Y_c - Y[1:, 1:]
    )
    areas[:, :, 2] = (Y[:-1, :-1] - Y_c) * (X_c - X[1:, :-1]) - (X[:-1, :-1] - X_c) * (
        Y_c - Y[1:, :-1]
    )
    areas[:, :, 3] = (Y[:-1, 1:] - Y_c) * (X_c - X[:-1, :-1]) - (X[:-1, 1:] - X_c) * (
        Y_c - Y[:-1, :-1]
    )

    inv_areas = flat_areas[:, :, None] / areas

    x_edges[:-1, :, 4] += inv_areas[:, :, 3]
    x_edges[1:, :, 4] -= inv_areas[:, :, 1]

    y_edges[:, 1:, 4] += inv_areas[:, :, 0]
    y_edges[:, :-1, 4] -= inv_areas[:, :, 2]

    internal_edges[:, :, :, 4] = inv_areas[:, :, [3, 0, 1, 2]] - inv_areas

    all_edges = np.concatenate(
        (x_edges.reshape(-1, 5), y_edges.reshape(-1, 5), internal_edges.reshape(-1, 5)),
        axis=0,
    )
    return all_edges, internal_edges


def clip_contribution(edges):
    contribution = (edges[:, 3] - edges[:, 1]) * edges[:, 4]
    skip = abs(contribution) < 1e-16

    if skip.sum() > 0:
        print(f"Skipping {skip.sum()} zero-effect edges")

    return edges[~skip]


def flip_edges(edges):
    edges = edges[:, [2, 3, 0, 1, 4]]
    edges[:, 4] *= -1
    return edges


def clip_to_x(edges, xlim):
    to_flip = edges[:, 0] > edges[:, 2]
    edges[to_flip, :] = flip_edges(edges[to_flip, :])
    # Can only skip right-hand edges, left-hand need to be handled more carefully
    skip = edges[:, 0] > xlim[1]
    if skip.sum() > 0:
        print(f"Skipping {skip.sum()} edges at right")
        edges = edges[~skip, :]

    skip_left = edges[:, 2] < xlim[0]
    # Pre-edge format: y0, y1, weight
    pre_edges_left = edges[skip_left][:, [1, 3, 4]]
    if skip_left.sum() > 0:
        print(f"Skipping {skip.sum()} edges at left")
        edges = edges[~skip_left, :]

    clip_right = edges[:, 2] > xlim[1]
    if clip_right.sum() > 0:
        print(f"Clipping {clip_right.sum()} edges at right")
        edges[clip_right, 3] = edges[clip_right, 1] + (
            edges[clip_right, 3] - edges[clip_right, 1]
        ) * (xlim[1] - edges[clip_right, 0]) / (
            edges[clip_right, 2] - edges[clip_right, 0]
        )
        edges[clip_right, 2] = xlim[1]

    clip_left = edges[:, 0] < xlim[0]
    pre_edges_clipped = edges[clip_left][:, [1, 3, 4]]
    if clip_left.sum() > 0:
        print(f"Clipping {clip_left.sum()} edges at left")
        y_intersect = edges[clip_left, 3] - (
            edges[clip_left, 3] - edges[clip_left, 1]
        ) * (edges[clip_left, 2] - xlim[0]) / (
            edges[clip_left, 2] - edges[clip_left, 0]
        )
        pre_edges_clipped[:, 1] = y_intersect
        edges[clip_left, 1] = y_intersect
        edges[clip_left, 0] = xlim[0]

    return np.concatenate((pre_edges_left, pre_edges_clipped), axis=0), edges


def clip_to_y(edges, ylim):
    to_flip = edges[:, 1] > edges[:, 3]
    edges[to_flip, :] = flip_edges(edges[to_flip, :])
    skip = (edges[:, 3] < ylim[0]) | (edges[:, 1] > ylim[1])
    if skip.sum() > 0:
        print(f"Skipping {skip.sum()} edges out of y lims")
        edges = edges[~skip, :]

    clip_top = edges[:, 3] > ylim[1]
    if clip_top.sum() > 0:
        print(f"Clipping {clip_top.sum()} edges at top")
        edges[clip_top, 2] = edges[clip_top, 0] + (
            edges[clip_top, 2] - edges[clip_top, 0]
        ) * (ylim[1] - edges[clip_top, 1]) / (edges[clip_top, 3] - edges[clip_top, 1])
        edges[clip_top, 3] = ylim[1]

    clip_bot = edges[:, 1] < ylim[0]
    if clip_bot.sum() > 0:
        print(f"Clipping {clip_bot.sum()} edges at bottom")
        edges[clip_bot, 0] = edges[clip_bot, 2] - (
            edges[clip_bot, 2] - edges[clip_bot, 0]
        ) * (edges[clip_bot, 3] - ylim[0]) / (edges[clip_bot, 3] - edges[clip_bot, 1])
        edges[clip_bot, 1] = ylim[0]

    return edges


@jit(nopython=True)
def fill_starting(start_values, pre_edge, Y_min):
    # Firstly flip the edge if it's 'downwards': exchange p0 <-> p1 and negate its weight
    y0, y1, dA = pre_edge
    if y1 < y0:
        y0, y1 = y1, y0
        dA *= -1

    all_y = np.arange(floor(y0), ceil(y1) + 1)

    all_y[0] = y0
    all_y[-1] = y1

    y_prevs = all_y[:-1]
    y_nexts = all_y[1:]

    for y_prev, y_next in zip(y_prevs, y_nexts):
        start_values[int(floor(y_prev) - Y_min)] += (y_next - y_prev) * dA


@jit(nopython=True)
def fill_accumulation(raster, edge, XY_min):
    # Firstly flip the edge if it's 'downwards': exchange p0 <-> p1 and negate its weight
    x0, y0, x1, y1, dA = edge
    if y1 < y0:
        x0, y0, x1, y1 = x1, y1, x0, y0
        dA *= -1

    islope = (x1 - x0) / (y1 - y0)

    all_y = np.arange(floor(y0), ceil(y1) + 1)
    all_x = x0 + (all_y - y0) * islope

    all_x[0] = x0
    all_x[-1] = x1
    all_y[0] = y0
    all_y[-1] = y1

    y_prevs = all_y[:-1]
    y_nexts = all_y[1:]

    x_min = all_x[:-1]
    x_max = all_x[1:]

    if islope < 0:
        x_min, x_max = x_max, x_min
        islope *= -1

    for y_prev, y_next, x_mns in zip(y_prevs, y_nexts, x_min):
        line = raster[:, int(floor(y_prev) - XY_min[1])]
        h = y_next - y_prev
        x_pls = x_mns + islope * h
        x_px_min = int(floor(x_mns))
        x_px_max = int(floor(x_pls)) + 1

        if (x_px_max - x_px_min) == 1:
            x_avg = (x_mns + x_pls) / 2 - x_px_min
            line[x_px_min - XY_min[0]] += (1 - x_avg) * h * dA
            line[x_px_max - XY_min[0]] += x_avg * h * dA
        else:
            slope = islope**-1
            dx0 = x_px_min - x_mns + 1
            dx1 = x_pls - x_px_max + 1
            line[x_px_min - XY_min[0]] += 1 / 2 * dx0 * dx0 * slope * dA
            line[x_px_max - XY_min[0]] += 1 / 2 * dx1 * dx1 * slope * dA
            line[np.arange(x_px_min + 1, x_px_max) - XY_min[0]] += slope * dA
            line[x_px_min + 1 - XY_min[0]] -= 1 / 2 * (1 - dx0) * (1 - dx0) * slope * dA
            line[x_px_max - 1 - XY_min[0]] -= 1 / 2 * (1 - dx1) * (1 - dx1) * slope * dA


def fill_accumulation_nojit(raster, edge, XY_min):
    # Firstly flip the edge if it's 'downwards': exchange p0 <-> p1 and negate its weight
    x0, y0, x1, y1, dA = edge
    if y1 < y0:
        x0, y0, x1, y1 = x1, y1, x0, y0
        dA *= -1

    islope = (x1 - x0) / (y1 - y0)

    all_y = np.arange(floor(y0), ceil(y1) + 1)
    all_x = x0 + (all_y - y0) * islope

    all_x[0] = x0
    all_x[-1] = x1
    all_y[0] = y0
    all_y[-1] = y1

    y_prevs = all_y[:-1]
    y_nexts = all_y[1:]

    x_min = all_x[:-1]
    x_max = all_x[1:]

    if islope < 0:
        x_min, x_max = x_max, x_min
        islope *= -1

    for y_prev, y_next, x_mns in zip(y_prevs, y_nexts, x_min):
        line = raster[:, int(floor(y_prev) - XY_min[1])]
        h = y_next - y_prev
        x_pls = x_mns + islope * h
        x_px_min = int(floor(x_mns))
        x_px_max = int(floor(x_pls)) + 1

        if (x_px_max - x_px_min) == 1:
            x_avg = (x_mns + x_pls) / 2 - x_px_min
            line[x_px_min - XY_min[0]] += (1 - x_avg) * h * dA
            line[x_px_max - XY_min[0]] += x_avg * h * dA
        else:
            slope = islope**-1
            dx0 = x_px_min - x_mns + 1
            dx1 = x_pls - x_px_max + 1
            line[x_px_min - XY_min[0]] += 1 / 2 * dx0 * dx0 * slope * dA
            line[x_px_max - XY_min[0]] += 1 / 2 * dx1 * dx1 * slope * dA
            line[np.arange(x_px_min + 1, x_px_max) - XY_min[0]] += slope * dA
            line[x_px_min + 1 - XY_min[0]] -= 1 / 2 * (1 - dx0) * (1 - dx0) * slope * dA
            line[x_px_max - 1 - XY_min[0]] -= 1 / 2 * (1 - dx1) * (1 - dx1) * slope * dA


def add_degenerate(raster, tri):
    pass
