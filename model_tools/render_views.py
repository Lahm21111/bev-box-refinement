"""Export annotation-free raster crops of the perturbed box for vision models."""
import argparse
import json
from pathlib import Path

from model_tools.geometry import EDGES, project, transform
import laspy
import matplotlib
import numpy as np
from PIL import Image, ImageDraw


def draw_box(draw, pixels, color, width, dashed=False):
    """Draw a wireframe; dashed previous edges remain distinct from solid current edges."""
    for i, j in EDGES:
        start, end = np.asarray(pixels[i]), np.asarray(pixels[j])
        if not dashed:
            draw.line([tuple(start), tuple(end)], fill=color, width=width)
            continue
        length = np.linalg.norm(end - start)
        if length == 0:
            continue
        direction = (end - start) / length
        for distance in np.arange(0, length, 5 * width):
            stop = min(distance + 3 * width, length)
            draw.line([tuple(start + distance * direction), tuple(start + stop * direction)],
                      fill=color, width=width)


def render_lidar(points, corners, bounds, axes, max_size, previous_corners=None, reference_view=None):
    u, v = axes
    if reference_view is not None:
        if reference_view['lidar_axes'] != list(axes):
            raise ValueError('Reference LiDAR axes differ from the requested view')
        scale = reference_view['pixels_per_meter']
        width, height = reference_view['size_wh']
        umin, vmax = reference_view['u_min_m'], reference_view['v_max_m']
    else:
        scale = (max_size - 1) / max(np.ptp(bounds[:, u]), np.ptp(bounds[:, v]))
        width, height = np.ceil((bounds[1] - bounds[0])[[u, v]] * scale).astype(int) + 1
        # Same number of pixels per meter horizontally and vertically, no stretching.
        umin, vmax = bounds[0, u], bounds[1, v]
    def pixels(xyz):
        return np.c_[(xyz[:, u] - umin) * scale, (vmax - xyz[:, v]) * scale]
    supersample = 3
    canvas = Image.new('RGB', (int(width) * supersample, int(height) * supersample), '#101c2c')
    draw = ImageDraw.Draw(canvas)
    height_norm = np.clip((points[:, 2] - bounds[0, 2]) / np.ptp(bounds[:, 2]), 0, 1)
    colors = (matplotlib.colormaps['winter'](height_norm)[:, :3] * 255).astype('uint8')
    for (x, y), color in zip(pixels(points) * supersample, colors):
        radius = 1.25 * supersample
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=tuple(color))
    box_pixels = pixels(corners)
    if previous_corners is not None:
        draw_box(draw, pixels(previous_corners) * supersample, '#ffd54f', 3 * supersample, dashed=True)
    draw_box(draw, box_pixels * supersample, '#ff604b', 3 * supersample)
    image = canvas.resize((int(width), int(height)), Image.Resampling.LANCZOS)
    mapping = dict(size_wh=[int(width), int(height)], lidar_axes=[int(u), int(v)],
                   pixels_per_meter=float(scale), u_min_m=float(umin), v_max_m=float(vmax),
                   lidar_to_pixel='px=(u-u_min_m)*pixels_per_meter; py=(v_max_m-v)*pixels_per_meter',
                   corners_pixels=box_pixels.tolist())
    return image, mapping


def render_views(source, offset, data_root, output, padding=.2, max_size=1024,
                 reference_mapping=None, previous_offset=None, view_frame=None):
    """Render the current 3D box as four clean images and return pixel mappings.

    This is the model-facing rendering tool; it does not generate random offsets.
    """
    source, offset, output = map(Path, (source, offset, output))
    data_root = Path(data_root) if data_root is not None else None
    if reference_mapping is not None:
        reference_mapping = Path(reference_mapping)
    if not np.isfinite(padding) or padding <= 0 or max_size < 64:
        raise ValueError('padding must be positive and max-size must be at least 64')
    meta = json.loads((source / 'metadata.json').read_text())
    frame = view_frame
    if frame is None and reference_mapping is not None:
        frame = json.loads(reference_mapping.read_text()).get('view_frame', 'lidar')
    if frame == 'box':
        from .box_views import render_box_views
        return render_box_views(source, offset, data_root, output, reference_mapping,
                                previous_offset, padding, max_size)
    if frame not in [None, 'lidar']:
        raise ValueError('Unknown view coordinate frame')
    if meta.get('dataset') == 'nuScenes':
        from .nuscenes_views import render_nuscenes_views
        return render_nuscenes_views(source, offset, data_root, output, padding, max_size,
                                     reference_mapping, previous_offset)
    data_root = data_root or Path('data/bench2drive')
    offset = json.loads(offset.read_text())
    for key in ['route', 'frame', 'box_id']:
        if meta[key] != offset[key]:
            raise ValueError(f'Source and offset disagree on {key}')
    corners = np.asarray(offset['perturbed_corners_lidar'])
    previous_corners = None
    if previous_offset is not None:
        previous = json.loads(Path(previous_offset).read_text())
        if any(previous[key] != offset[key] for key in ['route', 'frame', 'box_id']):
            raise ValueError('Previous box belongs to another sample')
        previous_corners = np.asarray(previous['perturbed_corners_lidar'])
        if previous_corners.shape != (8, 3) or not np.isfinite(previous_corners).all():
            raise ValueError('Previous box must have eight finite corners')
    # Include target surroundings so offset errors remain observable; draw only the shifted box.
    both = np.r_[np.asarray(offset['original_corners_lidar']), corners]
    if previous_corners is not None:
        both = np.r_[both, previous_corners]
    low, high = both.min(0), both.max(0)
    margin = np.maximum((high - low) * padding, .15)
    bounds = np.stack([low - margin, high + margin])
    reference = None
    if reference_mapping:
        reference = json.loads(reference_mapping.read_text())
        if (reference['frame'], reference['box_id']) != (meta['frame'], meta['box_id']):
            raise ValueError('Reference mapping belongs to another sample')
        bounds = np.asarray(reference['bounds_lidar'])
        if np.any(corners < bounds[0]) or np.any(corners > bounds[1]):
            raise ValueError('Perturbed box exceeds fixed reference crop')
        max_size = max(reference['views']['lidar_top']['size_wh'])
    route = data_root / meta['route']
    points = transform(np.asarray(laspy.read(route / 'lidar' / f'{meta["frame"]}.laz').xyz),
                       meta['ego_to_lidar'])
    points = points[np.all((points >= bounds[0]) & (points <= bounds[1]), axis=1)]
    output.mkdir(parents=True, exist_ok=True)
    mappings = {}
    for name, axes in [('lidar_top', (0, 1)), ('lidar_front', (1, 2)), ('lidar_side', (0, 2))]:
        image, mappings[name] = render_lidar(points, corners, bounds, axes, max_size, previous_corners,
                                             reference['views'][name] if reference else None)
        image.save(output / f'{name}.png')
    # Project the same perturbed 3D geometry into the camera.
    world = transform(corners, np.linalg.inv(np.asarray(meta['sensors']['LIDAR_TOP']['world2lidar'])))
    projected, depth = project(world, meta['sensors'][meta['camera']])
    if np.any(depth <= .1):
        raise ValueError('Perturbed box crosses camera near plane')
    camera_dir = 'rgb_' + meta['camera'].removeprefix('CAM_').lower()
    camera = Image.open(route / 'camera' / camera_dir / f'{meta["frame"]}.jpg').convert('RGB')
    original_pixels, _ = project(np.asarray(meta['annotation']['world_cord']),
                                 meta['sensors'][meta['camera']])
    union = np.r_[projected, original_pixels]
    if previous_corners is not None:
        previous_world = transform(previous_corners,
                                   np.linalg.inv(np.asarray(meta['sensors']['LIDAR_TOP']['world2lidar'])))
        previous_pixels, previous_depth = project(previous_world, meta['sensors'][meta['camera']])
        if np.any(previous_depth <= .1):
            raise ValueError('Previous box crosses camera near plane')
        union = np.r_[union, previous_pixels]
    low, high = union.min(0), union.max(0)
    padding = (high - low) * padding
    rect = np.r_[np.floor(np.maximum(low - padding, 0)),
                 np.ceil(np.minimum(high + padding, camera.size))].astype(int)
    if reference:
        rect = np.asarray(reference['views']['camera_crop']['crop_xyxy'])
    if np.any(rect[2:] <= rect[:2]):
        raise ValueError('Camera crop is outside image')
    draw = ImageDraw.Draw(camera)
    if previous_corners is not None:
        draw_box(draw, previous_pixels, '#ffd54f', 3, dashed=True)
    draw_box(draw, projected, '#ff604b', 3)
    camera.crop(tuple(rect)).save(output / 'camera_crop.png')
    mappings['camera_crop'] = dict(crop_xyxy=rect.tolist(),
                                    corners_pixels=(projected - rect[:2]).tolist(),
                                    size_wh=(rect[2:] - rect[:2]).tolist())
    # Geometry and labels live outside the model input PNGs.
    result = dict(frame=meta['frame'], box_id=meta['box_id'], offset=offset['offset'],
                  seed=offset['seed'], coordinates=meta['lidar_coordinates'],
                  bounds_lidar=bounds.tolist(), views=mappings,
                  image_content='only perturbed box and point cloud / camera RGB',
                  crop_policy=('fixed reference crop' if reference else
                               'union of original and perturbed box, padded; original box is not drawn'))
    if previous_corners is not None:
        result['image_content'] = 'previous box (yellow dashed), updated box (red solid), point cloud / RGB'
        result['previous_corners_lidar'] = previous_corners.tolist()
        result['box_styles'] = dict(previous='yellow dashed #ffd54f', updated='red solid #ff604b')
    (output / 'pixel_mapping.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('outputs/box_crops'))
    parser.add_argument('--offset', type=Path, default=Path('outputs/bev_random_offset/offset.json'))
    parser.add_argument('--data-root', type=Path, default=None)
    parser.add_argument('--output', type=Path, default=Path('outputs/offset_clean_views'))
    parser.add_argument('--padding', type=float, default=.2)
    parser.add_argument('--max-size', type=int, default=1024)
    parser.add_argument('--view-frame', choices=['box','lidar'], default=None)
    parser.add_argument('--previous-offset', type=Path, help='Overlay a previous box for human comparison')
    parser.add_argument('--reference-mapping', type=Path,
                        help='Keep the crop bounds and pixel scales of a previous export')
    args = parser.parse_args()
    if not np.isfinite(args.padding) or args.padding <= 0 or args.max_size < 64:
        parser.error('padding must be positive and max-size must be at least 64')
    result = render_views(**vars(args))
    print(json.dumps({name: mapping['size_wh'] for name, mapping in result['views'].items()}, indent=2))
    print(f'Saved to {args.output.resolve()}')


if __name__ == '__main__':
    main()
