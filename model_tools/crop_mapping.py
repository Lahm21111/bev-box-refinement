"""Expand rendering windows without changing per-view metric pixel scales."""
import copy
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import project, transform


def recover_corners(mapping):
    """Recover reference geometry from its recorded orthographic pixel projections."""
    corners = np.zeros((8, 3))
    assigned = set()
    for name in ['lidar_top', 'lidar_front', 'lidar_side']:
        view = mapping['views'][name]
        pixels = np.asarray(view['corners_pixels'])
        u, v = view['lidar_axes']
        for axis, values in [(u, pixels[:, 0]/view['pixels_per_meter'] + view['u_min_m']),
                             (v, view['v_max_m'] - pixels[:, 1]/view['pixels_per_meter'])]:
            if axis in assigned and not np.allclose(corners[:, axis], values, atol=1e-7):
                raise ValueError('Reference views disagree on box geometry')
            corners[:, axis] = values
            assigned.add(axis)
    if assigned != {0, 1, 2} or not np.isfinite(corners).all():
        raise ValueError('Reference is missing finite 3D corner geometry')
    if mapping.get('view_frame') == 'box':
        corners = transform(corners, mapping['view_to_lidar'])
    return corners


def expand_mapping(mapping, corners, meta, data_root=None, margin_px=24):
    """Return an expanded mapping and change flag; preserve each view's ppm."""
    result = copy.deepcopy(mapping)
    corners = np.asarray(corners)
    if corners.ndim != 2 or corners.shape[1] != 3 or not np.isfinite(corners).all():
        raise ValueError('Crop expansion requires finite XYZ points')
    bounds = np.asarray(mapping['bounds_lidar']).copy()
    ppm = mapping['views']['lidar_top']['pixels_per_meter']
    padding = margin_px / ppm
    low, high = corners.min(0), corners.max(0)
    bounds[0] = np.where(low < bounds[0], low-padding, bounds[0])
    bounds[1] = np.where(high > bounds[1], high+padding, bounds[1])
    changed = not np.array_equal(bounds, mapping['bounds_lidar'])
    result['bounds_lidar'] = bounds.tolist()
    for name in ['lidar_top', 'lidar_front', 'lidar_side']:
        view = result['views'][name]
        u, v = view['lidar_axes']
        scale = view['pixels_per_meter']
        # Keep unaffected views bit-for-bit unchanged, including any ceil-rounding.
        if np.array_equal(bounds[:, [u, v]], np.asarray(mapping['bounds_lidar'])[:, [u, v]]):
            continue
        view['u_min_m'], view['v_max_m'] = float(bounds[0, u]), float(bounds[1, v])
        size = np.ceil((bounds[1]-bounds[0])[[u,v]]*scale).astype(int)+1
        view['size_wh'] = size.tolist()
        view['corners_pixels'] = []  # Renderer writes the actual current corners.
    if meta.get('dataset') == 'nuScenes':
        from .nuscenes_views import project_corners
        pixels = project_corners(corners, meta)
        root = Path(data_root) if data_root is not None else Path(meta['data_root'])
        image_path = root / meta['camera_filename']
    else:
        world = transform(corners, np.linalg.inv(meta['sensors']['LIDAR_TOP']['world2lidar']))
        pixels, depth = project(world, meta['sensors'][meta['camera']])
        if np.any(depth <= .1):
            raise ValueError('Box crosses camera near plane')
        root = Path(data_root) if data_root is not None else Path('data/bench2drive')
        camera_dir = 'rgb_' + meta['camera'].removeprefix('CAM_').lower()
        image_path = root / meta['route'] / 'camera' / camera_dir / f'{meta["frame"]}.jpg'
    with Image.open(image_path) as image:
        size = np.asarray(image.size)
    rect = np.asarray(mapping['views']['camera_crop']['crop_xyxy'])
    low = np.maximum(np.floor(pixels.min(0)-margin_px), 0).astype(int)
    high = np.minimum(np.ceil(pixels.max(0)+margin_px), size).astype(int)
    # Expand only sides actually crossed; RGB remains bounded by the original photo.
    new_rect = np.r_[np.where(pixels.min(0)<rect[:2], np.minimum(rect[:2],low),rect[:2]),
                     np.where(pixels.max(0)>=rect[2:],np.maximum(rect[2:],high),rect[2:])]
    if not np.array_equal(rect, new_rect):
        changed = True
        result['views']['camera_crop'].update(crop_xyxy=new_rect.tolist(),
            size_wh=(new_rect[2:]-new_rect[:2]).tolist(), corners_pixels=[])
    return result, changed
