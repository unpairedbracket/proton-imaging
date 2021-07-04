import numpy as np

from numpy import floor, ceil

from numba import jit

def rasterize_grid(x0, y0, dx, dy, pixel_min, pixel_max, background_intensity=None):
    X = x0 + dx
    Y = y0 + dy

    orig_areas = np.diff(x0[:, :-1] + x0[:, 1:], axis=0) * np.diff(y0[:-1, :] + y0[1:, :], axis=1) / 8

    if background_intensity:
        X_c = (X[1:, 1:] + X[1:, :-1] + X[:-1, 1:] + X[:-1, :-1]) / 4
        Y_c = (Y[1:, 1:] + Y[1:, :-1] + Y[:-1, 1:] + Y[:-1, :-1]) / 4
        orig_areas *= background_intensity(X_c, Y_c)

    edges, degenerate_tris = get_edges(X, Y, orig_areas)

    X_min = int(ceil(X.min()) - 1); X_max = int(floor(X.max()) + 1)
    Y_min = int(ceil(Y.min()) - 1); Y_max = int(floor(Y.max()) + 1)
    raster = np.zeros((X_max - X_min + 1, Y_max - Y_min))
    contribution = abs((edges[:,3] - edges[:,1]) * edges[:,4])
    skipped = contribution < 1e-16
    
    if skipped.sum() > 0:
        print(f'Skipping {skipped.sum()} zero-effect edges')

    for edge in edges[~skipped, :]:
        fill_accumulation(raster, edge, (X_min, Y_min))

    for tri in degenerate_tris:
        add_degenerate(raster, tri)

    return raster, (X_min, X_max+1, Y_min, Y_max)


def get_edges(X, Y, flat_areas):
    quad_shape = X.shape
    X_c = (X[1:, 1:] + X[1:, :-1] + X[:-1, 1:] + X[:-1, :-1]) / 4
    Y_c = (Y[1:, 1:] + Y[1:, :-1] + Y[:-1, 1:] + Y[:-1, :-1]) / 4

    # Edge format: x_0, y_0, x_1, y_1, weight
    x_edges = np.zeros((quad_shape[0], quad_shape[1]-1, 5))
    y_edges = np.zeros((quad_shape[0]-1, quad_shape[1], 5))

    # Internal edge order (dim 2): NW, NE, SE, SW
    internal_edges = np.zeros((quad_shape[0]-1, quad_shape[1]-1, 4, 5))

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
    areas = np.zeros((quad_shape[0]-1, quad_shape[1]-1, 4))

    areas[:, :, 0] = (Y[1:, 1:] - Y_c) * (X_c - X[:-1, 1:]) - (X[1:, 1:] - X_c) * (Y_c - Y[:-1, 1:])
    areas[:, :, 1] = (Y[1:, :-1] - Y_c) * (X_c - X[1:, 1:]) - (X[1:, :-1] - X_c) * (Y_c - Y[1:, 1:])
    areas[:, :, 2] = (Y[:-1, :-1] - Y_c) * (X_c - X[1:, :-1]) - (X[:-1, :-1] - X_c) * (Y_c - Y[1:, :-1])
    areas[:, :, 3] = (Y[:-1, 1:] - Y_c) * (X_c - X[:-1, :-1]) - (X[:-1, 1:] - X_c) * (Y_c - Y[:-1, :-1])

    inv_areas = flat_areas[:, :, None] / areas

    x_edges[:-1, :, 4] += inv_areas[:, :, 3]
    x_edges[1:, :, 4] -= inv_areas[:, :, 1]

    y_edges[:, 1:, 4] += inv_areas[:, :, 0]
    y_edges[:, :-1, 4] -= inv_areas[:, :, 2]

    internal_edges[:, :, :, 4] = inv_areas[:, :, [3, 0, 1, 2]] - inv_areas

    all_edges = np.concatenate((x_edges.reshape(-1, 5), y_edges.reshape(-1, 5), internal_edges.reshape(-1, 5)), axis=0)
    return all_edges, internal_edges

@jit
def fill_accumulation(raster, edge, XY_min):
    # Firstly flip the edge if it's 'downwards': exchange p0 <-> p1 and negate its weight
    x0, y0, x1, y1, dA = edge
    if y1 < y0:
        x0, y0, x1, y1 = x1, y1, x0, y0
        dA *= -1

    islope = ((x1 - x0) / (y1 - y0))

    all_y = np.arange(floor(y0), ceil(y1)+1)
    all_x = x0 + (all_y - y0) * islope

    #all_x = np.concatenate(([x0], intermediate_x, [x1]))
    #all_y = np.concatenate(([y0], intermediate_y, [y1]))
    all_x[0] = x0; all_x[-1] = x1;
    all_y[0] = y0; all_y[-1] = y1;

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
            line[x_px_min - XY_min[0]] += 1/2 * dx0 * dx0 * slope * dA
            line[x_px_max - XY_min[0]] += 1/2 * dx1 * dx1 * slope * dA
            line[np.arange(x_px_min+1, x_px_max) - XY_min[0]] += slope * dA
            line[x_px_min + 1 - XY_min[0]] -= 1/2 * (1-dx0) * (1-dx0) * slope * dA
            line[x_px_max - 1 - XY_min[0]] -= 1/2 * (1-dx1) * (1-dx1) * slope * dA

def add_degenerate(raster, tri):
    pass
