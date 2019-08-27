"""Miscellaneous helper functions required for calculations
"""

import numpy as np

def calc_line_point_distance(line, point):
    """Function to calculate the distance of a point from a line

    Parameters
    ----------
    line : list
        (x1, y1, z1, x2, y2, z2) - the endpoints of the line segment
    point : list
        (x, y, z) - the point whose distance from line has to be calculcated

    Returns
    -------
    int
        The distance of the given point from the line represented by the
        list of endpoints
    """
    x1 = np.float32(np.asarray(line[:3]))
    x2 = np.float32(np.asarray(line[3:]))
    x = np.float32(np.asarray(point))

    numerator = np.linalg.norm(np.cross((x - x1), (x - x2)))
    denominator = np.linalg.norm((x2 - x1))

    dist = (numerator / denominator)
    #slope = ((line[1] - line[3]) / (line[0] - line[2]))
    #y_intercept = line[1] - slope * line[0]
    #b = -1.0
    #a = slope
    #return np.abs(a*point[0] + b*point[1] + y_intercept) / (np.sqrt(a**2 + b**2))

    return dist

def calc_distance(p1, p2):
    # p1 and p2 are 1D vectors
    p1 = np.float32(np.asarray(p1))
    p2 = np.float32(np.asarray(p2))
    return np.sqrt(np.sum(np.square(p1-p2)))

def reject_outliers(data, m=1.5):
    X=abs(data - np.mean(data, axis=0)) < 1.5 * np.std(data, axis=0)
    Y=np.logical_and(X[:,0],X[:,1],X[:,2])
    return data[Y, :]
