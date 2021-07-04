import numpy as np

def blit_tri(x0, y0, result=None, tri_value=1):
    '''
    x0, y0: array(3,) x and y positions of triangle corners in pixel units.
            The pixel grid is defined with corners at (xp, yp) for every integer xp and yp.
    Returns: array(n,m) representing the fractional overlap of the triangle with each pixel in its bounding box.
    '''
    ysort = np.argsort(y0)
    xsort = np.argsort(x0)
    x = x0[ysort] - np.floor(x0[xsort[0]])
    y = y0[ysort] - np.floor(y0[ysort[0]])
    xsort = np.argsort(x)

    x_p = np.arange(np.floor(x[xsort[2]]+2))
    y_p = np.arange(np.floor(y[2]+2))
    X, Y = np.meshgrid(x_p, y_p, indexing='ij')
    Xc = centre_mean(X)
    Yc = centre_mean(Y)

    if result is None:
        result = np.zeros_like(Xc)
    elif result.shape != Xc.shape:
        print(f'Specified result array wrong shape: {result.shape} should be {Xc.shape}')

    if result.shape == (1, 1):
        #print('degenerate triangle')
        result[0,0] += tri_value
        return result
    
    vert_scanlines = np.zeros((3, 3))
    x_intersect = x[0] + (y[1] - y[0]) * (x[2] - x[0]) / (y[2] - y[0])
    vert_scanlines[0, :] = y[0], x[0], x[0]
    vert_scanlines[1, :] = y[1], x[1], x_intersect
    vert_scanlines[2, :] = y[2], x[2], x[2]
    if y[0] == y[1]:
        vert_scanlines = vert_scanlines[1:,:]
    elif y[2] == y[1]:
        vert_scanlines = vert_scanlines[:-1,:]

    # calculate scanlines from edges intersecting the x-grid
    # This edge is the upper one
    jx0 = 1 + np.arange(*sorted((np.floor(x[1]), np.floor(x[2]))))
    scanlines_edge_0 = np.zeros((jx0.size, 3))
    scanlines_edge_0[:,0] = y[1] + (jx0 - x[1]) * (y[2] - y[1]) / (x[2] - x[1])
    scanlines_edge_0[:,1] = jx0
    scanlines_edge_0[:,2] = x[0] + (scanlines_edge_0[:,0] - y[0]) * (x[2] - x[0]) / (y[2] - y[0])

    # This is the one which hits the top and bottom vert
    jx1 = 1 + np.arange(*sorted((np.floor(x[2]), np.floor(x[0]))))
    scanlines_edge_1 = np.zeros((jx1.size, 3))
    scanlines_edge_1[:,0] = y[2] + (jx1 - x[2]) * (y[0] - y[2]) / (x[0] - x[2])
    scanlines_edge_1[:,1] = jx1
    top = scanlines_edge_1[:,0] > y[1]
    bot = scanlines_edge_1[:,0] < y[1]
    scanlines_edge_1[top,2] = x[1] + (scanlines_edge_1[top,0] - y[1]) * (x[2] - x[1]) / (y[2] - y[1])
    scanlines_edge_1[bot,2] = x[0] + (scanlines_edge_1[bot,0] - y[0]) * (x[1] - x[0]) / (y[1] - y[0])
    scanlines_edge_1 = scanlines_edge_1[top | bot, :]

    # This is the lower one
    jx2 = 1 + np.arange(*sorted((np.floor(x[0]), np.floor(x[1]))))
    scanlines_edge_2 = np.zeros((jx2.size, 3))
    scanlines_edge_2[:,0] = y[0] + (jx2 - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
    scanlines_edge_2[:,1] = jx2
    scanlines_edge_2[:,2] = x[0] + (scanlines_edge_2[:,0] - y[0]) * (x[2] - x[0]) / (y[2] - y[0])

    # Now do pixel ceilings and floors
    scanlines_pixel = np.zeros((y_p.size - 2, 3))
    scanlines_pixel[:,0] = y_p[1:-1]
    scanlines_pixel = scanlines_pixel[
            (scanlines_pixel[:,0] != y[0]) &
            (scanlines_pixel[:,0] != y[1]) &
            (scanlines_pixel[:,0] != y[2]), :
        ]
    scanlines_pixel[:,1] = x[0] + (scanlines_pixel[:,0] - y[0]) * (x[2] - x[0]) / (y[2] - y[0])

    top = scanlines_pixel[:,0] >= y[1]
    scanlines_pixel[ top,2] = x[1] + (scanlines_pixel[ top,0] - y[1]) * (x[2] - x[1]) / (y[2] - y[1])
    scanlines_pixel[~top,2] = x[0] + (scanlines_pixel[~top,0] - y[0]) * (x[1] - x[0]) / (y[1] - y[0])

    all_scanlines = np.concatenate((vert_scanlines, scanlines_edge_0, scanlines_edge_1, scanlines_edge_2, scanlines_pixel), axis=0)
    sort_scans = np.argsort(all_scanlines[:,0])
    sorted_scanlines = all_scanlines[sort_scans,:]
    sorted_scanlines[:,1:].sort()

    scanpoints = (sorted_scanlines[1:,:] + sorted_scanlines[:-1,:])/2
    Dy = np.diff(sorted_scanlines[:,0])
    
    partial_result = np.zeros_like(x_p[:-1])
    all_partial_results = np.zeros((partial_result.size, scanpoints.shape[0]))

    tri_area = abs(x[0] * y[1] - y[0] * x[1] + x[1] * y[2] - y[1] * x[2] + x[2] * y[0] - y[2] * x[0])/2
    tri_intensity = tri_value / tri_area

    for i, (point, dy) in enumerate(zip(scanpoints, Dy)):
        partial_result[:] = 0
        px = np.floor(point).astype(int)
        partial_result[px[1]:px[2]+1] = 1
        partial_result[px[1]] -= point[1] - px[1]
        partial_result[px[2]] -= px[2] + 1 - point[2]
        all_partial_results[:, i] = partial_result
        result[:, px[0]] += partial_result * dy * tri_intensity

    return result

def centre_mean(var):
    return (var[1:, 1:] + var[:-1,1:] + var[1:,:-1] + var[:-1,:-1]) / 4

def centre_prod(var):
    return (var[1:, 1:] * var[:-1,1:] * var[1:,:-1] * var[:-1,:-1])


