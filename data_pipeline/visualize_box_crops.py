"""Render one Bench2Drive annotation as three LiDAR crops and a camera crop."""
import argparse
import gzip
import json
from pathlib import Path

import laspy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from model_tools.geometry import EDGES, BOX_COLOR, transform, project

DEFAULT_ROUTE = 'AccidentTwoWays_Town13_Route30356_Weather10_02-10-03-31-29'


def draw_camera(image, pixels, width=3):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    for i, j in EDGES:
        draw.line([tuple(pixels[i]), tuple(pixels[j])], fill=BOX_COLOR, width=width)
    return result


def lidar_panel(ax, points, corners, axes, bounds, title):
    u, v = axes
    ax.set_facecolor('#101c2c')
    ax.scatter(points[:, u], points[:, v], c=points[:, 2], cmap='winter',
               s=5, vmin=bounds[0, 2], vmax=bounds[1, 2], linewidths=0)
    for i, j in EDGES:
        ax.plot(corners[[i, j], u], corners[[i, j], v], color=BOX_COLOR, lw=1.8)
    names = ['X forward (m)', 'Y right (m)', 'Z up (m)']
    ax.set(xlim=bounds[:, u], ylim=bounds[:, v], xlabel=names[u], ylabel=names[v])
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=.12)
    ax.set_title(title, loc='left', fontsize=12, pad=12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data/bench2drive'))
    parser.add_argument('--route', default=DEFAULT_ROUTE)
    parser.add_argument('--frame', type=int, default=100)
    parser.add_argument('--box-id', default='34598')
    parser.add_argument('--camera', default='CAM_FRONT')
    parser.add_argument('--padding', type=float, default=.2,
                        help='Extra fraction of box size on EACH side (default 20%%).')
    parser.add_argument('--output', type=Path, default=Path('outputs/box_crops'))
    args = parser.parse_args()
    if args.padding <= 0:
        parser.error('--padding must be positive')
    route = args.data_root / args.route
    stem = f'{args.frame:05d}'
    with gzip.open(route / 'anno' / f'{stem}.json.gz', 'rt') as stream:
        anno = json.load(stream)
    box = next((b for b in anno['bounding_boxes'] if str(b['id']) == args.box_id), None)
    if box is None:
        raise ValueError(f'Box {args.box_id} does not exist in frame {stem}')
    world = np.asarray(box['world_cord'])
    if world.shape != (8, 3):
        raise ValueError('Expected eight world-space box corners')
    lidar_transform = np.asarray(anno['sensors']['LIDAR_TOP']['world2lidar'])
    corners = transform(world, lidar_transform)
    # Bench2Drive data_collect.py saves LAZ in EGO coordinates after adding
    # the sensor mounting translation (-0.39, 0, 1.84), not in sensor space.
    ego = next(b for b in anno['bounding_boxes'] if b['class'] == 'ego_vehicle')
    ego_to_lidar = lidar_transform @ np.linalg.inv(np.asarray(ego['world2ego']))
    points = transform(np.asarray(laspy.read(route / 'lidar' / f'{stem}.laz').xyz),
                       ego_to_lidar)
    low, high = corners.min(0), corners.max(0)
    margin = np.maximum((high - low) * args.padding, .15)
    bounds = np.stack([low - margin, high + margin])
    crop = points[np.all((points >= bounds[0]) & (points <= bounds[1]), axis=1)]
    if len(crop) == 0:
        raise ValueError('No LiDAR points in crop; verify sensor coordinates')
    sensor = anno['sensors'][args.camera]
    pixels, depth = project(world, sensor)
    if np.any(depth <= .1):
        raise ValueError('Box crosses camera near plane; choose another box/camera')
    camera_dir = 'rgb_' + args.camera.removeprefix('CAM_').lower()
    image = Image.open(route / 'camera' / camera_dir / f'{stem}.jpg').convert('RGB')
    p0, p1 = pixels.min(0), pixels.max(0)
    pad = (p1 - p0) * args.padding
    crop_rect = np.r_[np.floor(np.maximum(p0 - pad, 0)),
                      np.ceil(np.minimum(p1 + pad, image.size))].astype(int)
    if np.any(crop_rect[2:] <= crop_rect[:2]):
        raise ValueError('Box outside camera image')
    args.output.mkdir(parents=True, exist_ok=True)
    camera_crop = draw_camera(image, pixels).crop(tuple(crop_rect))
    camera_crop.save(args.output / 'camera_crop.png')
    # Full image with crop outline makes the selected object's context reviewable.
    context = draw_camera(image, pixels)
    ImageDraw.Draw(context).rectangle(tuple(crop_rect), outline='#ffcc66', width=2)
    context.save(args.output / 'camera_context.jpg')
    plt.style.use('dark_background')
    views = [((0, 1), 'Top view | X-Y', 'lidar_top'),
             ((1, 2), 'Front view | Y-Z', 'lidar_front'),
             ((0, 2), 'Side view | X-Z', 'lidar_side')]
    for axes, title, filename in views:
        fig, ax = plt.subplots(figsize=(6, 4.5), layout='constrained')
        lidar_panel(ax, crop, corners, axes, bounds, title)
        fig.savefig(args.output / f'{filename}.png', dpi=180)
        plt.close(fig)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), layout='constrained')
    for ax, (axes, title, _) in zip(axs.flat, views):
        lidar_panel(ax, crop, corners, axes, bounds, title)
    axs[1, 1].imshow(camera_crop)
    axs[1, 1].axis('off')
    axs[1, 1].set_title(f'{args.camera} | projected 3D box', loc='left', fontsize=12)
    fig.suptitle(f'Bench2Drive | frame {stem} | object {args.box_id}\n'
                 f'{box["type_id"].split(".")[-1]}  ·  annotation box  ·  '
                 f'{len(crop)} local LiDAR points  ·  {args.padding:.0%} padding per side', fontsize=15)
    fig.savefig(args.output / 'overview.png', dpi=180)
    plt.close(fig)
    np.savez_compressed(args.output / 'geometry.npz', points=crop,
                        corners_lidar=corners, corners_world=world,
                        corners_camera_pixels=pixels, crop_bounds_lidar=bounds)
    metadata = dict(route=args.route, frame=stem, box_id=args.box_id,
                    box_source='dataset ground truth annotation', camera=args.camera,
                    object_type=box['type_id'], padding_per_side=args.padding,
                    lidar_coordinates='CARLA sensor: X forward, Y right, Z up; meters',
                    source_laz_coordinates='CARLA ego frame; transformed to LiDAR sensor frame',
                    ego_to_lidar=ego_to_lidar.tolist(),
                    lidar_crop='3D axis-aligned neighborhood of the box, shared by all views',
                    lidar_crop_bounds=bounds.tolist(), lidar_crop_points=len(crop),
                    annotation_num_points=box.get('num_points'),
                    camera_crop_xyxy=crop_rect.tolist(), annotation=box,
                    sensors={k: anno['sensors'][k] for k in ['LIDAR_TOP', args.camera]})
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps({k: metadata[k] for k in ['frame', 'box_id', 'lidar_crop_points',
                                              'annotation_num_points', 'camera_crop_xyxy']}, indent=2))
    print(f'Saved to {args.output.resolve()}')


if __name__ == '__main__':
    main()
