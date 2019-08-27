import matplotlib.pyplot as plt
import numpy as np
import os
import rdp

def angle(dir):
    """
    Returns the angles between vectors.

    Parameters:
    dir is a 2D-array of shape (N,M) representing N vectors in M-dimensional space.

    The return value is a 1D-array of values of shape (N-1,), with each value
    between 0 and pi.

    0 implies the vectors point in the same direction
    pi/2 implies the vectors are orthogonal
    pi implies the vectors point in opposite directions
    """
    dir2 = dir[1:]
    dir1 = dir[:-1]
    return np.arccos((dir1*dir2).sum(axis=1)/(
        np.sqrt((dir1**2).sum(axis=1)*(dir2**2).sum(axis=1))))

def getTurningPoints(points, use_rdp=False, tolerance=0.05, minAngle = np.pi/20):
    # tolerance = 0.05
    # min_angle = np.pi/20
    #filename = os.path.expanduser('aa.csv')
    #points = np.genfromtxt(filename,delimiter=",")

    if use_rdp:
        # Use the Ramer-Douglas-Peucker algorithm to simplify the path
        # http://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm
        # Python implementation: https://github.com/sebleier/RDP/
        mask = np.array(rdp.rdp(points.tolist(), tolerance, return_mask=True))
        simplified = points[mask]
    else:
        simplified = points

    # compute the direction vectors on the simplified curve
    directions = np.diff(simplified, axis=0)
    theta = angle(directions)
    # Select the index of the points with the greatest theta
    # Large theta is associated with greatest change in direction.
    turnIdx = np.where(theta>minAngle)[0]+1

    return simplified, turnIdx






# from rdp import rdp
# import numpy as np
# import matplotlib.pyplot as plt
# import os
#
# def angle(directions):
#     """Return the angle between vectors
#     """
#     vec2 = directions[1:]
#     vec1 = directions[:-1]
#
#     norm1 = np.sqrt((vec1 ** 2).sum(axis=1))
#     norm2 = np.sqrt((vec2 ** 2).sum(axis=1))
#     cos = (vec1 * vec2).sum(axis=1) / (norm1 * norm2)
#     return np.arccos(cos)
#
#
# tolerance = 1
# min_angle = np.pi*0.22
# filename = os.path.expanduser('aa.csv')
# points = np.genfromtxt(filename,delimiter=",")
# print(len(points))
# x, y = points.T
# # Build simplified (approximated) trajectory
# # using RDP algorithm.
# simplified_trajectory = rdp(points, epsilon=0.05)
# sx, sy = simplified_trajectory.T
#
# # Visualize trajectory and its simplified version.
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(x, y, 'r--', label='trajectory')
# ax.plot(sx, sy, 'b-', label='simplified trajectory')
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.legend(loc='best')
#
# # Define a minimum angle to treat change in direction
# # as significant (valuable turning point).
# min_angle = np.pi / 20.0
#
# # Compute the direction vectors on the simplified_trajectory.
# directions = np.diff(simplified_trajectory, axis=0)
# theta = angle(directions)
#
# # Select the index of the points with the greatest theta.
# # Large theta is associated with greatest change in direction.
# idx = np.where(theta > min_angle)[0] + 1
#
# # Visualize valuable turning points on the simplified trjectory.
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(sx, sy, 'gx-', label='simplified trajectory')
# ax.plot(sx[idx], sy[idx], 'ro', markersize = 7, label='turning points')
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.legend(loc='best')
# print('Done')