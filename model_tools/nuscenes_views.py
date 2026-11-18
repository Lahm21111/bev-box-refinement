"""Render updated nuScenes boxes using the saved sensor calibrations."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .geometry import transform


def sample_identity(meta):
    if meta.get('dataset') == 'nuScenes':
        return dict(dataset='nuScenes', route=f'nuScenes/{meta["scene"]}',
                    frame=meta['sample_token'], box_id=meta['box_token'])
    return {key: meta[key] for key in ['route', 'frame', 'box_id']}


def project_corners(corners, meta):
    world = transform(corners, meta['lidar_to_global'])
    optical = transform(world, meta['global_to_camera'])
    homogeneous = optical @ np.asarray(meta['camera_intrinsic']).T
    if np.any(optical[:, 2] <= .1):
        raise ValueError('Box crosses the nuScenes camera near plane')
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def render_nuscenes_views(source, offset, data_root, output, padding, max_size,
                          reference_mapping=None, previous_offset=None):
    from .render_views import render_lidar, draw_box
    source, output = Path(source), Path(output)
    meta = json.loads((source / 'metadata.json').read_text())
    state = json.loads(Path(offset).read_text())
    identity = sample_identity(meta)
    if any(state[key] != identity[key] for key in ['route', 'frame', 'box_id']):
        raise ValueError('nuScenes source and state belong to different targets')
    corners = np.asarray(state['perturbed_corners_lidar'])
    previous = None
    if previous_offset is not None:
        old = json.loads(Path(previous_offset).read_text())
        if any(old[key] != identity[key] for key in ['route', 'frame', 'box_id']):
            raise ValueError('Previous nuScenes box belongs to another target')
        previous = np.asarray(old['perturbed_corners_lidar'])
    for box in [corners] + ([previous] if previous is not None else []):
        if box.shape != (8, 3) or not np.isfinite(box).all():
            raise ValueError('nuScenes box must contain eight finite corners')
    union = np.r_[state['original_corners_lidar'], corners]
    if previous is not None:
        union = np.r_[union, previous]
    margin = np.maximum(np.ptp(union, axis=0)*padding, .15)
    bounds = np.array([union.min(0)-margin, union.max(0)+margin])
    reference = None
    if reference_mapping is not None:
        reference = json.loads(Path(reference_mapping).read_text())
        if (reference.get('dataset'), reference['frame'], reference['box_id']) != (
                'nuScenes', identity['frame'], identity['box_id']):
            raise ValueError('Reference belongs to another nuScenes sample')
        bounds = np.asarray(reference['bounds_lidar'])
        max_size = max(reference['views']['lidar_top']['size_wh'])
        if np.any(union < bounds[0]) or np.any(union > bounds[1]):
            raise ValueError('nuScenes box exceeds the fixed LiDAR crop')
    root = Path(data_root) if data_root is not None else Path(meta['data_root'])
    points = np.fromfile(root / meta['lidar_filename'], dtype=np.float32).reshape(-1, 5)[:, :3]
    crop = points[np.all((points >= bounds[0]) & (points <= bounds[1]), axis=1)]
    output.mkdir(parents=True, exist_ok=True)
    views = {}
    for name, axes in [('lidar_top', [0, 1]), ('lidar_front', [0, 2]), ('lidar_side', [1, 2])]:
        image, views[name] = render_lidar(crop, corners, bounds, axes, max_size, previous,
                                         reference['views'][name] if reference else None)
        image.save(output / f'{name}.png')
    pixels = project_corners(corners, meta)
    with Image.open(root / meta['camera_filename']) as original:
        camera = original.convert('RGB')
    projected_union = project_corners(union, meta)
    pad = np.ptp(projected_union, axis=0)*padding
    rect = np.r_[np.floor(np.maximum(projected_union.min(0)-pad, 0)),
                 np.ceil(np.minimum(projected_union.max(0)+pad, camera.size))].astype(int)
    if reference is not None:
        rect = np.asarray(reference['views']['camera_crop']['crop_xyxy'])
    if np.any(rect[2:] <= rect[:2]):
        raise ValueError('nuScenes camera crop is outside the image')
    draw = ImageDraw.Draw(camera)
    if previous is not None:
        draw_box(draw, project_corners(previous, meta), '#ffd54f', 3, dashed=True)
    draw_box(draw, pixels, '#ff604b', 3)
    camera.crop(tuple(rect)).save(output / 'camera_crop.png')
    views['camera_crop'] = dict(size_wh=(rect[2:]-rect[:2]).tolist(), crop_xyxy=rect.tolist(),
                                corners_pixels=(pixels-rect[:2]).tolist())
    result = dict(dataset='nuScenes', frame=identity['frame'], box_id=identity['box_id'],
        seed=state.get('seed', 0), offset=state['offset'], bounds_lidar=bounds.tolist(), views=views,
        coordinates=meta['coordinates'], image_content='only current red box and point cloud / camera RGB')
    if previous is not None:
        result.update(previous_corners_lidar=previous.tolist(),
            image_content='previous yellow dashed box, updated red solid box, point cloud / camera RGB')
    (output / 'pixel_mapping.json').write_text(json.dumps(result, indent=2))
    return result
