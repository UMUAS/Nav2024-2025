import numpy as np

#get unit direction vector from angles and elevation
def angles_to_direction(zaxis_rotate, elevation):
    zaxis_rotate = np.deg2rad(zaxis_rotate)
    elevation = np.deg2rad(elevation)
    x = np.cos(elevation) * np.cos(zaxis_rotate)
    y = np.cos(elevation) * np.sin(zaxis_rotate)
    z = np.sin(elevation)
    return np.array([x, y, z])


#find target point given points and angles
def triangulate_from_bearings(p_list, angles_list):
    points = []
    dirs = []

    #convert our angles to direction vectors
    for i in range(3):
        zaxis_rotate, elevation = angles_list[i]
        direction = angles_to_direction(zaxis_rotate, elevation)
        points.append(np.array(p_list[i]))
        dirs.append(direction)

    #make least square matrix
    A = []
    b = []
    for i in range(3):
        d = dirs[i].reshape(3, 1)
        I = np.identity(3)
        Pi = points[i].reshape(3, 1)
        A.append(I - d @ d.T)
        b.append((I - d @ d.T) @ Pi)

    A = np.sum(A, axis=0)
    b = np.sum(b, axis=0)

    target = np.linalg.lstsq(A, b, rcond=None)[0]
    return target.flatten()
