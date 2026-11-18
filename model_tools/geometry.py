"""Shared 3D transforms and camera projection, independent of dataset loading."""
import numpy as np

EDGES = [(i, i ^ bit) for i in range(8) for bit in (1, 2, 4) if i < (i ^ bit)]


BOX_COLOR = '#ff604b'


def transform(points, matrix):
    return np.c_[points, np.ones(len(points))] @ np.asarray(matrix).T[:, :3]


def project(world, sensor):
    # CARLA camera: X forward, Y right, Z up. Pinhole: X right, Y down, Z forward.
    xyz = transform(world, sensor['world2cam'])
    optical = xyz[:, [1, 2, 0]] * [1, -1, 1]
    pixels = optical @ np.asarray(sensor['intrinsic']).T
    return pixels[:, :2] / pixels[:, 2:3], optical[:, 2]


def perturb_box(corners, dx, dy, yaw_deg):
    """Rotate about box center around LiDAR +Z, then translate in LiDAR XY."""
    center = corners.mean(axis=0)
    angle = np.deg2rad(yaw_deg)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    return (corners - center) @ rotation.T + center + [dx, dy, 0]


def box_pose(corners):
    """Local box XYZ to LiDAR, for the project's binary-sign corner order."""
    corners = np.asarray(corners, dtype=float)
    if corners.shape != (8, 3) or not np.isfinite(corners).all():
        raise ValueError('Expected eight finite ordered box corners')
    edges = np.column_stack([corners[0]-corners[4], corners[0]-corners[2], corners[0]-corners[1]])
    lengths = np.linalg.norm(edges, axis=0)
    if np.min(lengths) < 1e-6:
        raise ValueError('Degenerate box dimensions')
    unit = edges / lengths
    if np.linalg.det(unit) < 0 or not np.allclose(unit.T @ unit, np.eye(3), atol=.01):
        raise ValueError('Box corners must define a right-handed cuboid in dataset order')
    u, _, vt = np.linalg.svd(unit)
    pose = np.eye(4)
    pose[:3, :3] = u @ vt
    pose[:3, 3] = corners.mean(0)
    return pose


def perturb_box_local(corners, dx, dy, yaw_deg, view_to_lidar):
    """Apply a translation and yaw expressed in the displayed box frame."""
    pose = np.asarray(view_to_lidar)
    basis = pose[:3, :3]
    center = np.asarray(corners).mean(0)
    angle = np.deg2rad(yaw_deg)
    rz = np.array([[np.cos(angle),-np.sin(angle),0], [np.sin(angle),np.cos(angle),0], [0,0,1]])
    rotation = basis @ rz @ basis.T
    return (corners-center) @ rotation.T + center + basis @ [dx,dy,0]
